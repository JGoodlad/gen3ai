"""This module defines a base class for communicating with showdown servers."""

import asyncio
import collections
import json
import logging
from logging import Logger
from time import perf_counter
from typing import Awaitable, Callable, List, Optional, Set

import ssl
import urllib.parse

import requests
import websockets as ws
from websockets import ClientConnection
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from poke_env.concurrency import (
    POKE_LOOP,
    create_in_poke_loop,
    handle_threaded_coroutines,
)
from poke_env.exceptions import ShowdownException
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration

BattleMessageCallback = Callable[[List[List[str]]], Awaitable[None]]
ChallengeCallback = Callable[[List[str]], Awaitable[None]]
QueryResponseCallback = Callable[[List[str]], Awaitable[None]]


async def _noop_battle_message(split_messages: List[List[str]]) -> None:
    return None


async def _noop_challenge_message(split_message: List[str]) -> None:
    return None


class PSClient:
    """
    Pokemon Showdown client.

    Responsible for communicating with showdown servers. Also implements some higher
    level methods for basic tasks, such as changing avatar and low-level message
    handling.
    """

    def __init__(
        self,
        account_configuration: AccountConfiguration,
        *,
        avatar: Optional[str] = None,
        log_level: Optional[int] = None,
        on_battle_message: Optional[BattleMessageCallback] = None,
        on_update_challenges: Optional[ChallengeCallback] = None,
        on_challenge_request: Optional[ChallengeCallback] = None,
        on_query_response: Optional[QueryResponseCallback] = None,
        server_configuration: ServerConfiguration,
        start_listening: bool = True,
        open_timeout: Optional[float] = 10.0,
        ping_interval: Optional[float] = 20.0,
        ping_timeout: Optional[float] = 20.0,
        proxy_url: Optional[str] = None,
        loop: asyncio.AbstractEventLoop = POKE_LOOP,
    ):
        """
        :param account_configuration: Account configuration.
        :type account_configuration: AccountConfiguration
        :param avatar: Account avatar name. Optional.
        :type avatar: str, optional
        :param log_level: The client's logger level.
        :type log_level: int. Defaults to logging's default level.
        :param server_configuration: Server configuration.
        :type server_configuration: ServerConfiguration
        :param start_listening: Whether to start listening to the server. Defaults to
            True.
        :type start_listening: bool
        :param open_timeout: How long to wait for a timeout when connecting the socket
            (important for backend websockets.
            Increase only if timeouts occur during runtime).
            If None connect will never time out.
        :type open_timeout: float, optional
        :param ping_interval: How long between keepalive pings (Important for backend
            websockets). If None, disables keepalive entirely.
        :type ping_interval: float, optional
        :param ping_timeout: How long to wait for a timeout of a specific ping
            (important for backend websockets.
            Increase only if timeouts occur during runtime).
            If None pings will never time out.
        :type ping_timeout: float, optional
        """
        self._active_tasks: Set[asyncio.Task] = set()
        self._battle_locks: dict[str, asyncio.Lock] = {}
        self._open_timeout = open_timeout
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._proxy_url = proxy_url

        self._server_configuration = server_configuration
        self._account_configuration = account_configuration

        self._avatar = avatar
        self._on_battle_message = on_battle_message or _noop_battle_message
        self._on_update_challenges = on_update_challenges or _noop_challenge_message
        self._on_challenge_request = on_challenge_request or _noop_challenge_message
        self._on_query_response = on_query_response or _noop_challenge_message

        self.loop = loop
        self._logged_in: asyncio.Event = create_in_poke_loop(asyncio.Event, loop)
        # Set when listen() exits for any reason WE did not request — an abnormal drop
        # (ping timeout, server died, no close frame) OR a clean peer/server-initiated
        # close (e.g. the server stopping). The connection layer is the only thing that
        # knows the socket is gone; it announces it here so a consumer blocked on an
        # _AsyncQueue (env.step/reset) fails loudly instead of hanging forever on a
        # message that will never come.
        self._disconnected: asyncio.Event = create_in_poke_loop(asyncio.Event, loop)
        # True once WE initiate the teardown (stop_listening) or the loop is cancelled,
        # so listen()'s exit handler can tell an intentional close — exit clean, do NOT
        # signal _disconnected — from a close we did not ask for. Terminating the
        # connection on purpose is not an error.
        self._closing: bool = False
        self._sending_lock: asyncio.Lock = create_in_poke_loop(asyncio.Lock, loop)

        self.websocket: ClientConnection
        self._logger: Logger = self._create_logger(log_level)

        if start_listening:
            self._listening_coroutine = asyncio.run_coroutine_threadsafe(
                self.listen(), self.loop
            )

    async def _handle_battle_message(self, split_messages: List[List[str]]) -> None:
        await self._on_battle_message(split_messages)

    async def _update_challenges(self, split_message: List[str]) -> None:
        await self._on_update_challenges(split_message)

    async def _handle_challenge_request(self, split_message: List[str]) -> None:
        await self._on_challenge_request(split_message)

    async def accept_challenge(self, username: str, packed_team: Optional[str]):
        assert self.logged_in.is_set(), f"Expected {self.username} to be logged in."
        await self.set_team(packed_team)
        await self.send_message("/accept %s" % username)

    async def challenge(self, username: str, format_: str, packed_team: Optional[str]):
        assert self.logged_in.is_set(), f"Expected {self.username} to be logged in."
        await self.set_team(packed_team)
        await self.send_message(f"/challenge {username}, {format_}")

    def _create_logger(self, log_level: Optional[int]) -> Logger:
        """Creates a logger for the client.

        Returns a named Logger that propagates to the root handler.
        Only sets the level if explicitly requested; otherwise inherits from root.

        :param log_level: The logger's level.
        :type log_level: int
        :return: The logger.
        :rtype: Logger
        """
        logger = logging.getLogger(self.username)
        if log_level is not None:
            logger.setLevel(log_level)
        # Do not add a StreamHandler here — let the application configure the root
        # logger however it wants (Rich capture, basicConfig, etc.).
        return logger

    async def _handle_message(self, message: str):
        """Handle received messages.
        :param message: The message to parse.
        :type message: str
        """
        try:
            # Showdown protocol: lines starting with '>' specify the room for subsequent lines.
            # Lines before any '>' or after an empty '>' belong to the global room.
            room_messages = collections.defaultdict(list)
            current_room = ""
            
            for line in message.split("\n"):
                if not line: continue
                if line.startswith(">"):
                    current_room = line[1:]
                
                split_line = line.split("|")
                
                # Check for "Always Global" commands that should never be handled by a battle room
                if len(split_line) > 1 and split_line[1] in {
                    "pm", "challstr", "updateuser", "updatechallenges", "queryresponse", "updatesearch"
                }:
                    room_messages[""].append(split_line)
                else:
                    room_messages[current_room].append(split_line)

            for room, lines in room_messages.items():
                if room.startswith("battle-"):
                    battle_tag = room
                    if battle_tag not in self._battle_locks:
                        self._battle_locks[battle_tag] = asyncio.Lock()
                    async with self._battle_locks[battle_tag]:
                        await self._handle_battle_message(lines)
                    
                    # Check for deinit in any of the lines
                    if any("deinit" in "|".join(l) for l in lines):
                        self._battle_locks.pop(battle_tag, None)
                
                else:
                    # Global messages or other rooms
                    for split_message in lines:
                        if len(split_message) <= 1: continue
                        msg_type = split_message[1]
                        
                        if msg_type == "challstr":
                            await self.log_in(split_message)
                        elif msg_type == "updateuser":
                            received_name = split_message[2].strip()
                            if received_name in [self.username, self.username + "@!"]:
                                self.logged_in.set()
                            elif received_name.startswith("Guest ") and not self._account_configuration.password:
                                self.logged_in.set()
                        elif "updatechallenges" in msg_type:
                            await self._update_challenges(split_message)
                        elif msg_type == "queryresponse":
                            await self._on_query_response(split_message)
                        elif msg_type == "pm":
                            if len(split_message) > 4 and split_message[4].startswith("/challenge"):
                                await self._handle_challenge_request(split_message)
                            else:
                                self.logger.info("Received PM: %s", "|".join(split_message))
                        elif msg_type == "nametaken":
                            self.logger.warning(
                                "Name '%s' rejected (%s) — continuing as guest",
                                split_message[2] if len(split_message) > 2 else "?",
                                split_message[3] if len(split_message) > 3 else "",
                            )
                            self.logged_in.set()
                        elif msg_type == "popup":
                            self.logger.warning("Popup message received: %s", "|".join(split_message))

        except Exception as exception:
            self.logger.exception("Unhandled exception in _handle_message:\n%s", message)
            raise exception

    async def _stop_listening(self):
        # We initiated this close — mark it so listen()'s exit handler treats it as a
        # clean teardown (no _disconnected signal, no crash).
        self._closing = True
        await self.websocket.close()

    async def change_avatar(self, avatar_name: Optional[str]):
        """Changes the account's avatar.

        :param avatar_name: The new avatar name. If None, nothing happens.
        :type avatar_name: int
        """
        await self.wait_for_login()
        if avatar_name is not None:
            await self.send_message(f"/avatar {avatar_name}")

    async def listen(self):
        """Listen to a showdown websocket and dispatch messages to be handled."""
        self.logger.info("Starting listening to showdown websocket")
        try:
            if self._proxy_url:
                from python_socks.async_.asyncio import Proxy
                parsed = urllib.parse.urlparse(self.websocket_url)
                dest_port = parsed.port or (443 if parsed.scheme == "wss" else 80)
                # python-socks doesn't recognise socks5h:// — normalise to socks5://.
                # Remote hostname resolution still happens by default when a hostname is passed.
                proxy_url = self._proxy_url.replace("socks5h://", "socks5://", 1)
                proxy = Proxy.from_url(proxy_url)
                sock = await proxy.connect(dest_host=parsed.hostname, dest_port=dest_port)
                ssl_ctx = ssl.create_default_context() if parsed.scheme == "wss" else None
                connect_cm = ws.connect(
                    self.websocket_url, sock=sock, ssl=ssl_ctx,
                    max_queue=None, open_timeout=self._open_timeout,
                    ping_interval=self._ping_interval, ping_timeout=self._ping_timeout,
                )
            else:
                connect_cm = ws.connect(
                    self.websocket_url,
                    max_queue=None, open_timeout=self._open_timeout,
                    ping_interval=self._ping_interval, ping_timeout=self._ping_timeout,
                )
            async with connect_cm as websocket:
                self.websocket = websocket
                while True:
                    try:
                        message = await websocket.recv()
                    except ws.exceptions.ConnectionClosedOK:
                        break
                    
                    message_str = str(message)
                    
                    # --- SMART FLUSH ---
                    # If this frame contains a request, suck up all PENDING frames immediately.
                    # We use a 1ms timeout to ensure we get everything currently in the buffer
                    # without causing a long delay.
                    if "|request|" in message_str:
                        while True:
                            try:
                                # Non-blocking read: return immediately if no data is buffered
                                extra_message = await asyncio.wait_for(websocket.recv(), timeout=0)
                                message_str += "\n" + str(extra_message)
                            except (asyncio.TimeoutError, ws.exceptions.ConnectionClosedOK):
                                break

                    self.logger.debug("\033[92m\033[1m<<<\033[0m %s", message_str)
                    task = asyncio.create_task(self._handle_message(message_str))
                    self._active_tasks.add(task)
                    task.add_done_callback(self._active_tasks.discard)

        except ConnectionClosedOK:
            self.logger.warning(
                "Websocket connection with %s closed", self.websocket_url
            )
        except (asyncio.CancelledError, RuntimeError) as e:
            # Loop teardown / task cancellation — an intentional shutdown, NOT a crash.
            self._closing = True
            self.logger.critical("Listen interrupted by %s", e)
        except Exception as e:
            # Abnormal drop (e.g. ConnectionClosedError "no close frame received or
            # sent", ping timeout, server died). Logged here; the finally announces it.
            self.logger.exception(e)
        finally:
            # listen() is exiting, so the message pump is dead: any consumer blocked on
            # an _AsyncQueue.get() (env.step/reset) is waiting for a message that can
            # never arrive. Wake them so they raise ShowdownException and the process
            # exits (→ launcher restarts from checkpoint) instead of hanging forever.
            # EXCEPT when the teardown is OURS — stop_listening(), or the loop
            # cancellation above, sets _closing — because a clean self-requested close
            # is expected and signalling it would turn a healthy exit into a spurious
            # crash-restart. This is the case the old "clean ConnectionClosedOK is not
            # signalled" rule protected; the flag keeps that protection while still
            # catching a close we did NOT ask for (server stopped, peer dropped us).
            if not self._closing:
                self._disconnected.set()

    async def log_in(self, split_message: List[str]):
        """Log in with specified username and password.

        Split message contains information sent by the server. This information is
        necessary to log in.

        :param split_message: Message received from the server that triggers logging in.
        :type split_message: List[str]
        """
        if self.account_configuration.password:
            proxies = (
                {"https": self._proxy_url, "http": self._proxy_url}
                if self._proxy_url else None
            )
            log_in_request = requests.post(
                self.server_configuration.authentication_url,
                data={
                    "act": "login",
                    "name": self.account_configuration.username,
                    "pass": self.account_configuration.password,
                    "challstr": split_message[2] + "%7C" + split_message[3],
                },
                timeout=10.0,
                proxies=proxies,
            )
            self.logger.info("Sending authentication request")
            assertion = json.loads(log_in_request.text[1:])["assertion"]
        else:
            self.logger.info("Bypassing authentication request")
            assertion = ""

        await self.send_message(f"/trn {self.username},0,{assertion}")

        await self.change_avatar(self._avatar)

    async def search_ladder_game(self, format_: str, packed_team: Optional[str]):
        await self.set_team(packed_team)
        await self.send_message(f"/search {format_}")

    async def send_message(
        self, message: str, room: str = "", message_2: Optional[str] = None
    ):
        """Sends a message to the specified room.

        `message_2` can be used to send a sequence of length 2.

        :param message: The message to send.
        :type message: str
        :param room: The room to which the message should be sent.
        :type room: str
        :param message_2: Second element of the sequence to be sent. Optional.
        :type message_2: str, optional
        """
        if message_2:
            to_send = "|".join([room, message, message_2])
        else:
            to_send = "|".join([room, message])
        self.logger.debug("\033[93m\033[1m>>>\033[0m %s", to_send)
        await self.websocket.send(to_send)

    async def set_team(self, packed_team: Optional[str]):
        if packed_team:
            await self.send_message(f"/utm {packed_team}")
        else:
            await self.send_message("/utm null")

    async def stop_listening(self):
        await handle_threaded_coroutines(self._stop_listening(), self.loop)

    async def wait_for_login(self, checking_interval: float = 0.001, wait_for: int = 5):
        start = perf_counter()
        while perf_counter() - start < wait_for:
            await asyncio.sleep(checking_interval)
            if self.logged_in.is_set():
                return
        assert self.logged_in.is_set(), f"Expected {self.username} to be logged in."

    @property
    def account_configuration(self) -> AccountConfiguration:
        """The client's account configuration.

        :return: The client's account configuration.
        :rtype: AccountConfiguration
        """
        return self._account_configuration

    @property
    def logged_in(self) -> asyncio.Event:
        """Event object associated with user login.

        :return: The logged-in event
        :rtype: Event
        """
        return self._logged_in

    @property
    def logger(self) -> Logger:
        """Logger associated with the client.

        :return: The logger.
        :rtype: Logger
        """
        return self._logger

    @property
    def server_configuration(self) -> ServerConfiguration:
        """The client's server configuration.

        :return: The client's server configuration.
        :rtype: ServerConfiguration
        """
        return self._server_configuration

    @property
    def username(self) -> str:
        """The account's username.

        :return: The account's username.
        :rtype: str
        """
        return self.account_configuration.username

    @property
    def websocket_url(self) -> str:
        """The websocket url.

        It is derived from the server url.

        :return: The websocket url.
        :rtype: str
        """
        return self.server_configuration.websocket_url
