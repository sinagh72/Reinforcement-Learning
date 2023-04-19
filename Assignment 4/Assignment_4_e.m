% Let's assume the data was generated before
total_Episode = 50000;
episode_Length = 35;
state = nan(total_Episode, episode_Length);
action = nan(total_Episode, episode_Length);
R = nan(total_Episode, episode_Length);

for episode = 1 : total_Episode
    state(episode, 1) = randi(25);

    for t = 1 : episode_Length
        action(episode, t) = policy(state(episode, t));
        [next_state, reward] = environment(state(episode, t), action(episode, t));
        state(episode, t+1) = next_state;
        R(episode, t) = reward;

    end

end

save("states.mat","state");
save("actions.mat","action");
save("rewards.mat","R");

%% Load the data and run it
discountFactor = 0.99;
total_Episode = 50000;
episode_Length = 35;

load('states.mat')
load('actions.mat')
load('rewards.mat')

b = 0.2*ones(25, 5); % MC policy
b(13,:) = [zeros(1,4) 1];
b(14,:) = b(13,:);
b(15,:) = b(14,:);

[new_policy, Q] = MC_policy_prediction(state, action, R, discountFactor, b, total_Episode, episode_Length);
string_policy = convert(new_policy);

function [target_policy, Q] = MC_policy_prediction(states, actions, rewards, gamma, b, Total_Episode, Episode_length)
    Q = -10*ones(25,5);
    % Q
    C = zeros(25, 5);
    
    [~,idx] = max(Q,[], 2);
    target_policy = (idx == 1:5); %target policy

    for episode = 1:Total_Episode
        G = 0;
        W = 1;

        for t = Episode_length:-1:1
            s = states(episode,t);
            a = actions(episode, t);
            G = G*gamma + rewards(episode, t);
            C(s, a) = C(s, a) + W;
            Q(s, a) = Q(s, a) + (W/(C(s, a)))*(G - Q(s, a));
            [~, a_max] = max(Q(s,:));
            target_policy(s,:) = zeros(1,5);
            target_policy(s, a_max) = 1;
            W = W*target_policy(s,a)/b(s,a);
            if W == 0
                break
            end
            
        end
    end
end

function out = convert(policy)
     [~,idx] = max(policy,[], 2);
     for i = 1:length(idx)
         if idx(i) == 1
             out(i) = "up";
         elseif idx(i) == 2
            out(i) = "down";
         elseif idx(i) == 3
            out(i) = "left";
         elseif idx(i) == 4
            out(i) = "right";
         else
            out(i) = "stay";
         end
     end
end
