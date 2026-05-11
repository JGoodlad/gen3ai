from poke_env.battle.side_condition import SideCondition

def test_spikes():
    print(f"SideCondition.SPIKES: {SideCondition.SPIKES}")
    print(f"Type: {type(SideCondition.SPIKES)}")
    print(f"Name: {SideCondition.SPIKES.name}")
    
    test_dict = {SideCondition.SPIKES: 1}
    print(f"Lookup with 'spikes': {test_dict.get('spikes', 'NOT FOUND')}")
    print(f"Lookup with SideCondition.SPIKES: {test_dict.get(SideCondition.SPIKES, 'NOT FOUND')}")
    
    # Check all SideCondition names
    print("All SideConditions:")
    for sc in SideCondition:
        print(f" - {sc.name}: {sc}")

if __name__ == "__main__":
    test_spikes()
