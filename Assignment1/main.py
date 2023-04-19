def transition(s, a):
    if a == "red":
        return s
    elif a == "blue":
        return str(max(0, int(s) - 4))
    elif a == "black":
        return str(min(5, int(s) + 1))


if __name__ == "__main__":
    t = 10
    g_terminal = {"0": 0, "1": 3, "2": 4, "3": 6, "4": 6, "5": 10}
    actions = ["blue", "red", "black"]
    states = g_terminal.keys()
    # state action reward
    g = {"0": {"blue": 3, "red": 2, "black": 0},
         "1": {"blue": 4, "red": 2, "black": 0},
         "2": {"blue": 5, "red": 2, "black": 0},
         "3": {"blue": 6, "red": 2, "black": 0},
         "4": {"blue": 7, "red": 2, "black": 0},
         "5": {"blue": 7, "red": 2, "black": 2}, }
    V = [float('inf')] * (t + 2)
    V[-1] = {"0": 0, "1": 3, "2": 4, "3": 6, "4": 6, "5": 10}
    print(V)
    for t in range(t, -1, -1):
        print("Time: ", t)
        V[t] = {}
        for s in states:
            possibles = {}
            for a in actions:
                # v(s) = g(s,a) + v(s')
                s_prime = transition(s, a)
                possibles[a] = g[s][a] + V[t + 1][s_prime]
                print(f"Q_{t}({s},{a}) = g(s,a) + V(s') = {g[s][a]} + {V[t + 1][s_prime]} = {possibles[a]},"
                      f" s': ({s_prime})")
            a_min = min(possibles, key=possibles.get)
            V[t][s] = possibles[a_min]
            print("=========")
            print(f"optimal action: {a_min}")
            print(f"V_{t}({s}) = min[g(s,a) + V(s')] = {V[t][s]}")
            print("=========")
        print("==========================================")
    print(V[10])
