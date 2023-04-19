clear; clc;

%% Main Program:

% Action_To_Char (3)

Counter = 0;
Counter_2 = 0;
State_Action_Reward{25,5} =[];
Discount_Factor = 0.99;

Epsilon = 0.15;

% ==== Followed_Policy Initializing ===============================
for i = 1:1:25
    
    %     Followed_Policy{i} =  Policy_Function(i);
    Followed_Policy{i} =  'Up';
    
    
end
% =================================================================
Actions_All = {'Up','Left','Down','Right','Stay'};

% ----------- Initialization for Q: ---------------
for i = 1:1:25
    for j = 1:1:5
        Q_State_Action(i,j) = 0;
    end
end
% -------------------------------------------------

% i = Episode Number // t = Experiment Number
Total_Episodes = 1000;
Episodes_Length = 15;

for i = 1:1:Total_Episodes
    
    % ----------- Pick random first action based on criterias : ----------
    State_0 = randi(25);
    if ( ismember([State_0] , [13 14 15]) )
        Action_0 = 'Stay';
    else
        Action_0 = Actions_All{ randi(5) };
    end
    % --------------------------------------------------------------------
    
    States_Arr {i,1} = State_0;
    Actions_Arr {i,1} = Action_0;
%     Rewards_Arr = {};
    
    State = State_0;
    Action = Action_0;
    
%     Transition_Function (State, Action)
    
    % Generating Episode:
    for T = 1:1:Episodes_Length
        
        [Applied_Action, State_Prime, Reward] = Transition_Function (State, Action);
        
        
        States_Arr {i,T} = State;        
        Actions_Arr {i,T} = Applied_Action;
        
        Rewards_Arr {i,T} = Reward;
        
        State = State_Prime;
        Action = Followed_Policy (State);
%         Action = Policy_Function (State);

               
    end
    
    State_Action_Episode = { States_Arr{i,:} ; Actions_Arr{i,:} };
    
    Q_First_Visit_Condition= zeros(25,5);
     action = randsample(1:5,1,false, Followed_Policy{State})
    % --------------------------------------------------------------------
    G = 0;
    for t = Episodes_Length:-1:1
        G = Discount_Factor * G + Rewards_Arr {i,t};
        
        if ( Q_First_Visit_Condition( States_Arr{i,t} ,...
                Action_To_Number( Actions_Arr{i,t} ) ) == 0 )
            
            Q_First_Visit_Condition( States_Arr{i,t} ,...
                Action_To_Number( Actions_Arr{i,t} )) = 1;
            
            State_Action_Reward{ States_Arr{i,t} ,...
                Action_To_Number( Actions_Arr{i,t} ) } = ...
                [ State_Action_Reward{ States_Arr{i,t} ,...
                Action_To_Number( Actions_Arr{i,t} ) } ; G ];
            
            Q_State_Action(States_Arr{i,t} ,...
                Action_To_Number( Actions_Arr{i,t} )) = ...
                mean (State_Action_Reward{ States_Arr{i,t} ,...
                Action_To_Number( Actions_Arr{i,t} ) });
            
            % ============================================================
            
%             Q_Compare = Q_State_Action(States_Arr{i,t} ,...
%                 Action_To_Number( Actions_Arr{i,t} ));
%     
            % ============================================================
            % Selecting the best action for applying on policy
            Q_Max = max ( Q_State_Action(States_Arr{i,t} , 1:1:5  ) );
%             
            for k = 1:1:5
                if ( Q_State_Action(States_Arr{i,t}, k) == Q_Max)   
                    Action_Star = k;
                end
            end
%             
            % Updating the Polciy:
            Policy_Probablity = rand;
            if ( Policy_Probablity < (1-Epsilon) )
                Followed_Policy{States_Arr{i,t}} = Action_To_Char( Action_Star ) ;
            else
                Followed_Policy{States_Arr{i,t}} = Actions_All{randi(5)};
            end
            
%           % ============================================================
        end        
                
    end
    % --------------------------------------------------------------------
%  disp('Stop');   
    
end



%% Policy Function Definition:
function [Action] = Policy_Function (State)
    
    if (State == 13 || State == 14 || State == 15)
        
        Action = 'Stay';
    
    else
        
        Prob_Dist = 100*rand;
        if (Prob_Dist >= 0 && Prob_Dist < 20)
            Action = 'Up';
        elseif (Prob_Dist >= 20 && Prob_Dist < 40)
            Action = 'Left';
        elseif (Prob_Dist >= 40 && Prob_Dist < 60)
            Action = 'Down';
        elseif (Prob_Dist >= 60 && Prob_Dist < 80)
            Action = 'Right';
        elseif (Prob_Dist >= 80 && Prob_Dist < 100)
            Action = 'Stay';                        
        end
            
    end
    
end


%% Transition Function Definition:


% function [State_prime] = Transition_Function (State, Action);
function [Applied_Action, State_Prime, Reward] = Transition_Function (State, Action)

    State_Column = mod(State, 5);
    if ( State_Column == 0 )
        State_Column = 5;
    end;
    State_Row = ( (State - State_Column) / 5 ) + 1;

%     disp('Row :'); disp(State_Row);    
%     disp('Column :'); disp(State_Column);    
    
    Prob_Dist = 100*rand;
    
%     Applied_Action
    
    if ( strcmp(Action,'Stay') )
        Applied_Action = Action;
    end
    
    if ( strcmp(Action,'Up') )

        if (Prob_Dist >= 0 && Prob_Dist < 70)
            Applied_Action = 'Up';         
        elseif(Prob_Dist >= 70 && Prob_Dist < 80)
            Applied_Action = 'Left';            
        elseif(Prob_Dist >= 80 && Prob_Dist < 90)
            Applied_Action = 'Right';            
        elseif(Prob_Dist >= 90 && Prob_Dist < 100)
            Applied_Action = 'Stay';
            
        end               
    end
    
    if ( strcmp(Action,'Left') )

        if (Prob_Dist >= 0 && Prob_Dist < 70)
            Applied_Action = 'Left';         
        elseif(Prob_Dist >= 70 && Prob_Dist < 80)
            Applied_Action = 'Down';            
        elseif(Prob_Dist >= 80 && Prob_Dist < 90)
            Applied_Action = 'Up';            
        elseif(Prob_Dist >= 90 && Prob_Dist < 100)
            Applied_Action = 'Stay';
            
        end               
    end
    
    if ( strcmp(Action,'Down') )

        if (Prob_Dist >= 0 && Prob_Dist < 70)
            Applied_Action = 'Down';         
        elseif(Prob_Dist >= 70 && Prob_Dist < 80)
            Applied_Action = 'Right';            
        elseif(Prob_Dist >= 80 && Prob_Dist < 90)
            Applied_Action = 'Left';            
        elseif(Prob_Dist >= 90 && Prob_Dist < 100)
            Applied_Action = 'Stay';
            
        end               
    end
    
    if ( strcmp(Action,'Right') )

        if (Prob_Dist >= 0 && Prob_Dist < 70)
            Applied_Action = 'Right';         
        elseif(Prob_Dist >= 70 && Prob_Dist < 80)
            Applied_Action = 'Up';            
        elseif(Prob_Dist >= 80 && Prob_Dist < 90)
            Applied_Action = 'Down';            
        elseif(Prob_Dist >= 90 && Prob_Dist < 100)
            Applied_Action = 'Stay';
            
        end               
    end
    
    
    if ( strcmp(Applied_Action,'Stay') )
       State_Prime = State; 
    end
    
    if ( strcmp(Applied_Action,'Up') )
        if ( State_Row == 5 )
            State_Prime = State;
        else
            State_Prime = State_Row * 5 + State_Column;
        end 
    end
    
    if ( strcmp(Applied_Action,'Left') ) 
        if ( State_Column == 1 )
            State_Prime = State;
        else
            State_Prime = (State_Row - 1 )* 5 + (State_Column - 1);
        end 
    end
    
    if ( strcmp(Applied_Action,'Down') ) 
        if ( State_Row == 1 )
            State_Prime = State;
        else
            State_Prime = (State_Row - 2) * 5 + State_Column;
        end 
    end
    
    if ( strcmp(Applied_Action,'Right') )
        if ( State_Column == 5 )
            State_Prime = State;
        else
            State_Prime = (State_Row - 1) * 5 + (State_Column + 1);
        end 
    end

    R = -100;
    
    if ( ismember([State_Prime] , [3 4 5]) )
        Reward = R;
    elseif ( ismember([State_Prime] , [13 14 15]) )
        Reward = 0;
    
    else
        
        if ( strcmp(Applied_Action, 'Stay') )
            Reward = -0.5;
        elseif ( strcmp(Action, Applied_Action) && State ~= State_Prime )
            Reward = -1;
        else
            Reward = -0.5;
        end
        
    end
        
            

end

function [Action_Number] = Action_To_Number (Action)

    if ( strcmp(Action, 'Up') )
        Action_Number = 1;
    end
    if ( strcmp(Action, 'Left') )
        Action_Number = 2;
    end
    if ( strcmp(Action, 'Down') )
        Action_Number = 3;
    end
    if ( strcmp(Action, 'Right') )
        Action_Number = 4;
    end
    if ( strcmp(Action, 'Stay') )
        Action_Number = 5;
    end

end


function [Action_Char] = Action_To_Char (Action)

    if (Action==1)
        Action_Char = 'Up';
    elseif (Action==2)
        Action_Char = 'Left';
    elseif (Action==3)
        Action_Char = 'Down';
    elseif (Action==4)
        Action_Char = 'Right';
    elseif (Action==5)
        Action_Char = 'Stay';
    end
    
end




