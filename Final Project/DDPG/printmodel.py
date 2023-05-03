from model import ActorNet, CriticNet
actor_main_net = ActorNet(n_states=24, n_actions=4, seed=0)
critic_main_net = CriticNet(n_states=24, n_actions=4, seed=0)

print(actor_main_net)
print(critic_main_net)
