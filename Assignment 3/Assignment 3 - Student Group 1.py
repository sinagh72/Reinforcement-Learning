import time

BOARD_ROWS = 5
BOARD_COLS = 5
ACTIONS = ["up", "down", "left", "right", "stay"]
REWARDS = [-1, -0.5, 0]
STATES = []
# creating the states
for i in range(BOARD_COLS):
    for j in range(BOARD_ROWS):
        STATES += [(i, j)]


def nxt_possible_states(s, a):
    """
    a: up, down, left, right, stay
    -------------
    4 |
    3 |
    2 |
    1 |
    0 | 1 | 2 | 3 | 4 |
    :return a list of tuples (row, col, flag). flag indicates that the state is the desired or not
    """
    s_primes = []
    if a == "up":
        s_primes += [(s[0] + 1, s[1], 1)]
        s_primes += [(s[0], s[1] - 1, 0)]
        s_primes += [(s[0], s[1] + 1, 0)]
        s_primes += [(s[0], s[1], 0)]

    elif a == "down":
        s_primes += [(s[0] - 1, s[1], 1)]
        s_primes += [(s[0], s[1] - 1, 0)]
        s_primes += [(s[0], s[1] + 1, 0)]
        s_primes += [(s[0], s[1], 0)]

    elif a == "left":
        s_primes += [(s[0], s[1] - 1, 1)]
        s_primes += [(s[0] + 1, s[1], 0)]
        s_primes += [(s[0] - 1, s[1], 0)]
        s_primes += [(s[0], s[1], 0)]

    elif a == "right":
        s_primes += [(s[0], s[1] + 1, 1)]
        s_primes += [(s[0] + 1, s[1], 0)]
        s_primes += [(s[0] - 1, s[1], 0)]
        s_primes += [(s[0], s[1], 0)]

    elif a == "stay":
        s_primes += [(s[0], s[1], 1)]

    for i in range(0, len(s_primes)):
        if s_primes[i][0] < 0 or s_primes[i][0] >= BOARD_ROWS or s_primes[i][1] < 0 or s_primes[i][1] >= BOARD_COLS:
            s_primes[i] = (s[0], s[1], 0)  # invalid s' becomes s with 0 flag

    return s_primes


def s_prime_given_s_a(s_prime, s, a):  # P(s' | s, a)
    prob = 0
    s_primes = nxt_possible_states(s, a)  # get all the possible next states (neighbor list)
    if s_prime not in [(ss[0], ss[1]) for ss in s_primes]:  # check whether it's possible to go to s' from s given a
        return 0  # impossible move from s to s'
    if a == "stay":
        prob = 1
    else:
        if s_prime == s_primes[0][0:2]:  # if element of the neighbor list is same is as s' (first element has 0.7 prob)
            prob += 0.7

        for ss in s_primes[1:]:  # check orthogonal states
            if s_prime == ss[0:2]:
                prob += 0.1
    return prob


def r_given_s_a_s_prime(r, s, a, s_prime, R):
    """
    :param r: reward
    :param s: state
    :param a: action
    :param s_prime: s'
    :param R: input reward R
    :return: P(r, s' | s, a)
    """
    s_primes = nxt_possible_states(s, a)
    if s_prime not in [(ss[0], ss[1]) for ss in s_primes]:  # impossible move from s to s'
        return 0

    if s_prime[0] == 2 and s_prime[1] >= 2:  # in [(2, 2), (2, 3), (2, 4)]
        if r == 0:
            return 1
        else:
            return 0

    elif s_prime[0] == 0 and s_prime[1] >= 2:  # in [(0, 2), (0, 3), (0, 4)]
        if r == R:
            return 1
        else:
            return 0

    elif len(s_primes) == 1:  # if stay action was taken
        if r == -0.5:
            return 1
        else:
            return 0

    elif s_prime == s_primes[0][0:2] and s_primes[0][2] == 1:  # if desired state was reached
        if r == -1:
            return 1
        else:
            return 0
    else:
        if r == -0.5:
            return 1
        else:
            return 0


def update_policy(R, policy, V):
    """

    :param R: input reward R
    :param policy: current policy
    :param V: value function
    :return: updated policy
    """
    for s in STATES:
        policy_val = {}
        for a in ACTIONS:
            policy[s][a] = 0
            val = 0
            for r in REWARDS:
                for s_prime in STATES:
                    val += (r + V[s_prime]) * r_given_s_a_s_prime(r=r, s=s, a=a, s_prime=s_prime, R=R) * \
                           s_prime_given_s_a(s_prime=s_prime, s=s, a=a)
            policy_val[a] = val
        policy[s][max(policy_val, key=policy_val.get)] = 1  # P(a | s) = 1 for arg max V(s) else P(a | s) = 0


def update_value_function(R, N, V, policy):
    """

    :param R: input reward R
    :param N: number of iterations
    :param policy: current policy
    :param V: value function
    :return: updated value function
    """
    V_new = V.copy()
    for i in range(N):
        for s in STATES:
            val = 0
            for a in ACTIONS:
                for r in REWARDS:
                    for s_prime in STATES:
                        val += (r + V[s_prime]) * r_given_s_a_s_prime(r=r, s=s, a=a, s_prime=s_prime, R=R) * \
                               s_prime_given_s_a(s=s, a=a, s_prime=s_prime) * policy[s][a]
            V_new[s] = val
    return V_new


def policy_value_iterations(R, policy, V, N, theta):
    counter = 0
    while True:
        update_policy(R, V=V, policy=policy)  # update the policy
        V_new = update_value_function(R, N, V, policy)  # update the value function
        counter += 1
        difference_val = {k: abs(V_new[k] - V[k]) for k in V.keys()}  # find the |V_old - V_new|
        if max(difference_val.values()) < theta:  # terminate condition
            break
        V = V_new.copy()

    return policy, V, counter


def printing_policy(policy):
    output = ""
    for s in reversed(list(policy)):
        for a in policy[s]:
            if policy[s][a] == 1:
                output = "{0:2d}".format(s[0] * BOARD_ROWS + s[1] + 1) + " : " + "{0:5s}".format(a) + " | " + output
        if (s[0] * BOARD_ROWS + s[1] + 1) % 5 == 1:
            print(output)
            output = ""


def printing_value(V):
    output = ""
    for s in reversed(list(V)):
        output = "{0:2d}".format(s[0] * BOARD_ROWS + s[1] + 1) + " : " + "{0:10f}".format(V[s]) + " | " + output
        if (s[0] * BOARD_ROWS + s[1] + 1) % 5 == 1:
            print(output)
            output = ""


def run(R, N, flag):
    policy = {}
    V = {}
    for s in STATES:
        V[s] = 0
        policy[s] = {}
        for a in ACTIONS:
            policy[s][a] = 0.2
    print(f"================================= R = {R} ==================================")
    starting_time = time.time()
    policy, V, counter = policy_value_iterations(R=R, policy=policy, V=V, N=N, theta=theta)
    print("Time elapsed (in seconds):", time.time() - starting_time)
    if flag:
        print(f"================================ N = {N} ==================================")
        print("Total iterations:", counter)
    print("==============================Policy========================================")
    printing_policy(policy)
    print("===============================Value Function===============================")
    printing_value(V)
    print("============================================================================")


if __name__ == "__main__":
    theta = 1e-10
    N = 10
    R = [-1, -100, -0.05]
    for r in R:
        flag = False
        if r not in REWARDS:
            flag = True
            REWARDS.append(r)
        run(r, N, False)
        if flag:
            REWARDS.remove(r)
    N = [1, 5, 10, 100]
    REWARDS = [-1, -0.5, 0, -100]
    for n in N:
        run(R=-100, N=n, flag=True)
