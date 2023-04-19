function [next_state, reward] = environment(state, action)

% Generates Probability that action will do what it supposed to
r = rand(1);

% Reward for reaching states 3, 4, 5
R = -100;

% UP action is chosen by policy
if action == 1
    if r <= 0.7
        next_state = state + 5;
        if state >= 21
            next_state = state;
        end
    
    elseif (0.7 < r) && (r <= 0.8)
        next_state = state + 1;
        if mod(state,5) == 0
            next_state = state;
        end
    
    elseif (0.8 < r) && (r <= 0.9)
        next_state = state - 1;
        if mod(state,5) == 1
            next_state = state;
        end

    else
        next_state = state;
    end

% DOWN action is chosen by policy
elseif action == 2
    if r <= 0.7
        next_state = state - 5;
        if state <= 5
            next_state = state;
        end
    
    elseif (0.7 < r) && (r <= 0.8)
        next_state = state + 1;
        if mod(state,5) == 0
            next_state = state;
        end
    
    elseif (0.8 < r) && (r <= 0.9)
        next_state = state - 1;
        if mod(state,5) == 1
            next_state = state;
        end

    else
        next_state = state;
    end

% LEFT action is chosen by policy
elseif action == 3
    if r <= 0.7
        next_state = state - 1;
        if mod(state,5) == 1
            next_state = state;
        end
    
    elseif (0.7 < r) && (r <= 0.8)
        next_state = state + 5;
        if state >= 21
            next_state = state;
        end
    
    elseif (0.8 < r) && (r <= 0.9)
        next_state = state - 5;
        if state <= 5
            next_state = state;
        end

    else
        next_state = state;
    end

% RIGHT action is chosen by policy
elseif action == 4
    if r <= 0.7
        next_state = state + 1;
        if mod(state,5) == 0
            next_state = state;
        end
    
    elseif (0.7 < r) && (r <= 0.8)
        next_state = state + 5;
        if state >= 21
            next_state = state;
        end
    
    elseif (0.8 < r) && (r <= 0.9)
        next_state = state - 5;
        if state <= 5
            next_state = state;
        end

    else
        next_state = state;
    end

% STAY action is chosen by policy
else
    next_state = state;

end


% Rewards when new state equals 3, 4, 5, 13, 14, 15
if (next_state == 3) || (next_state == 4) || (next_state == 5)
    reward = R;

elseif (next_state == 13) || (next_state == 14) || (next_state == 15)
    reward = 0;
% Rewards when new state does not equal 3, 4, 5, 13, 14, 15
elseif (action == 1 && next_state == state + 5) || ...
        (action == 2 && next_state == state - 5) || ...
        (action == 3 && next_state == state - 1) || ...
        (action == 4 && next_state == state + 1)
%     || ...
%         (action == 5 && next_state == state)
    reward = -1;
else
    reward = -0.5;
end

end

