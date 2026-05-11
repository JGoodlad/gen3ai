from poke_env.battle.effect import Effect

def test_lookup():
    volatiles = {Effect.CONFUSION: 1, Effect.SUBSTITUTE: 1}
    
    print(f"volatiles keys: {list(volatiles.keys())}")
    print(f"'confusion' in volatiles: {'confusion' in volatiles}")
    print(f"Effect.CONFUSION in volatiles: {Effect.CONFUSION in volatiles}")
    
    # Check if there's any string normalization in poke_env's Effect
    try:
        print(f"Effect('confusion'): {Effect('confusion')}")
    except Exception as e:
        print(f"Effect('confusion') failed: {e}")

if __name__ == "__main__":
    test_lookup()
