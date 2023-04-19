%% Data generation step
total_Episode = 100;
episode_Length = 35;
discountFactor = 0.99;

state = nan(total_Episode, episode_Length);
action = nan(total_Episode, episode_Length);
R = nan(total_Episode, episode_Length);

for episode = 1 : total_Episode
    state(episode, 1) = 1;

    for t = 1 : episode_Length
        action(episode, t) = policy(state(episode, t));
        [next_state, reward] = environment(state(episode, t), action(episode, t));
        state(episode, t+1) = next_state;
        R(episode, t) = (discountFactor^(t-1))*reward;

    end

end
%%

%% policy optimization
mu_optimal = zeros(25, 5);
Q = Q_learning(0.5, discountFactor, 0.1, state, action, total_Episode, episode_Length)


function Q = Q_learning(alpha, gamma, epsilon, states, actions, Total_Episodes, Episode_Length)
    %initialize the Q value
    Q = zeros(25, 5);
    for episode = 1:Total_Episodes
        for t = Episode_Length - 1:1
            s = state(episode, t);
            action = epsilon_greedy(Q, s, epsilon);
            R(episode, t)
            [~, best_next_action] = max(Q(next_state,:));
            Q(state, action) = Q(state, action) + alpha*(reward+gamma*(Q(next_state, best_next_action) - Q(state, action)));
        end
    end
end