from poke_env.battle.effect import Effect

def test_volatiles():
    print("All Effects:")
    for e in Effect:
        print(f" - {e.name}: {e}")

if __name__ == "__main__":
    test_volatiles()
