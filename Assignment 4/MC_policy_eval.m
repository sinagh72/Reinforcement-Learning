clearvars;

%% Part A

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

%% Part B

State_Reward{25} = [];

for episode = 1 : total_Episode 
    visited = false(25,1);
    episode_length = sum(~isnan(R(episode, :)));
    for t = 1 : episode_Length
        s = state(episode, t);
        if  ~visited(s) 
            State_Reward{s}(end+1) = sum(R(episode, t:episode_length));
            visited(s) = true;
        end
    end
end

 