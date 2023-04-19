function a = policy(state)

if (state == 13) || (state == 14) || (state == 15)
    a = 5;

else
    r = rand;
    
    if r <= 0.2 
        a = 1;
    
    elseif (0.2 < r) && (r <= 0.4)
        a = 2;

    elseif (0.4 < r) && (r <= 0.6)
        a = 3;
    
    elseif (0.6 < r) && (r <= 0.8)
        a = 4;

    elseif r > 0.8
        a = 5;

    end
end

end
