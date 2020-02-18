classdef PFSA
    %   Class File Containing the Essential PFSA Functions needed
    %   Code written by Chandrachur Bhattacharya @ Penn State
    %   Date Last Modified: January, 2020
    %   Functions In-built and their syntax
    %   (For details please type: PFSA/function_name)
    %
    %   FUNCTIONS IMPORTANT TO THE USER
    %
    %   [arr] = chooseNfromP(obj,N,P)
    %   [bounds] = return_partitions_unit_var(obj,x,t,n)
    %   [bounds] = return_partitions_unit_range(obj,x,t,n)
    %
    %   [y] = normalize_unit_var(obj,x)
    %   [y] = normalize_unit_range(obj,x)
    %   [bounds] = return_partitions_unit_var(obj,x,t,n)
    %   [bounds] = return_partitions_unit_range(obj,x,t,n)
    %   [A] = State_Transition_Matrix(obj,y,n,D,states)
    %   [A] = State_Transition_Matrix_D1(obj,y,n)
    %   [y] = symbolize_partitions_unit_range(obj,x,bounds)
    %   [y] = symbolize_partitions_unit_var(obj,x,bounds)
    %   [y] = symbolize_unit_var(obj,x,t,n)
    %   [y] = symbolize_unit_range(obj,x,t,n)
    %   [states] = generate_states(obj,n,D)
    %   [P] = State_Probability_Vector(obj,y,n,D,states)
    
    methods
        function [arr] = chooseNfromP(obj,N,P)
            %   Code written by Chandrachur Bhattacharya @ Penn State
            %   Date Last Modified: November 25, 2019
            %   FORMAT: [arr] = chooseNfromP(obj,N,P)
            %   PURPOSE: This function chooses N numbers from P numbers randomly
            %   INPUTS: N = No. of Numbers to be chosen,
            %           P = Total Numbers
            %   OUPUTS: arr = Array of N selected numbers
            %   MATLAB functions called: ceil isempty intersect
            %   Non-MATLAB functions called: None
            
            n = 1;
            arr = [];
            
            while n<=N
                temp = ceil(rand*P);
                if isempty(intersect(temp,arr))==1
                    arr(n) = temp;
                    n = n+1;
                end
            end
        end
        
        function [y] = normalize_unit_var(obj,x)
            %   Code written by Chandrachur Bhattacharya @ Penn State
            %   Date Last Modified: November 25, 2019
            %   FORMAT: [y] = normalize_unit_var(obj,x)
            %   PURPOSE: This function returns the normalized time-series to have zero
            %            mean and unit variance
            %   INPUTS: x = Raw time-series
            %   OUPUTS: y =  Normalized time-series
            %   MATLAB functions called: mean std
            %   Non-MATLAB functions called: None
            y = (x-mean(x))/std(x);
        end
        
        function [y] = normalize_unit_range(obj,x)
            %   Code written by Chandrachur Bhattacharya @ Penn State
            %   Date Last Modified: November 25, 2019
            %   FORMAT: [y] = normalize_unit_range(obj,x)
            %   PURPOSE: This function returns the normalized time-series
            %            to have unit range
            %   INPUTS: x = Raw time-series
            %   OUPUTS: y =  Normalized time-series
            %   MATLAB functions called: max min
            %   Non-MATLAB functions called: None
            range = max(x) - min(x);
            y = (x-min(x))/range;
        end
        
        function [bounds] = return_partitions_unit_var(obj,x,t,n)
            %   Code written by Chandrachur Bhattacharya @ Penn State
            %   Date Last Modified: November 25, 2019
            %   FORMAT: [bounds] = return_partitions_unit_var(obj,x,t,n)
            %   PURPOSE: This function returns the partition boundaries for a given raw
            %            data time series after normalizing the data to have zero mean
            %            and unit variance
            %   INPUTS: x = Raw time-series,
            %           t = Type of partitioning schemes (MEP: 'm', Uniform: 'u',
            %               K-means: 'k'),
            %           n = Alphabet size
            %   OUPUTS: bounds = normalized partition boundary values
            %   MATLAB functions called: min max sort size kmeans find
            %   Non-MATLAB functions called: normalizeTS
            
            x = normalize_unit_var(PFSA,x);
            bounds = zeros(1,n);
            
            switch t
                
                case 'u'
                    bounds = [0:1/n:1]*(max(x) - min(x)) + min(x);
                    
                case 'm'
                    x1 = sort(x);
                    K = max(size(x1));
                    bounds(1) = min(x1);
                    for i=1:n
                        bounds(i+1) = x1(ceil(i*K/n));
                    end
                    
                case 'k'
                    y = kmeans(x',n,'MaxIter',10000);
                    bounds(1) = min(x);
                    for i=1:n
                        bounds(i+1) = max(x(find(y == i)));
                    end
                    bounds = sort(bounds);
            end
        end
        
        function [bounds] = return_partitions_unit_range(obj,x,t,n)
            %   Code written by Chandrachur Bhattacharya @ Penn State
            %   Date Last Modified: November 25, 2019
            %   FORMAT: [bounds] = return_partitions_unit_range(obj,x,t,n)
            %   PURPOSE: This function returns the partition boundaries for a given raw
            %            data time series after normalizing the data to have zero mean
            %            and unit variance
            %   INPUTS: x = Raw time-series,
            %           t = Type of partitioning schemes (MEP: 'm', Uniform: 'u',
            %               K-means: 'k'),
            %           n = Alphabet size
            %   OUPUTS: bounds = normalized partition boundary values
            %   MATLAB functions called: min max sort size kmeans find
            %   Non-MATLAB functions called: normalizeTS
            
            x = normalize_unit_range(PFSA,x);
            bounds = zeros(1,n);
            
            switch t
                
                case 'u'
                    bounds = [0:1/n:1]*(max(x) - min(x)) + min(x);
                    
                case 'm'
                    x1 = sort(x);
                    K = max(size(x1));
                    bounds(1) = min(x1);
                    for i=1:n
                        bounds(i+1) = x1(ceil(i*K/n));
                    end
                    
                case 'k'
                    y = kmeans(x',n,'MaxIter',10000);
                    bounds(1) = min(x);
                    for i=1:n
                        bounds(i+1) = max(x(find(y == i)));
                    end
                    bounds = sort(bounds);
            end
        end
        
        function [A] = State_Transition_Matrix(obj,y,n,D,states)
            %   Code written by Chandrachur Bhattacharya @ Penn State
            %   Date Last Modified: November 25, 2019
            %   FORMAT: [A] = State_Transition_Matrix(obj,y,n,D,states)
            %   PURPOSE: This function returns the state transition matrix for the PFSA
            %            given a symbol sequence, Markov depth, alphabet size and the
            %            list of states
            %   INPUTS: y = Symbol sequence,
            %           n = Alphabet size,
            %           D = Markov depth,
            %           states = List of states (can be generated by function
            %           'generate_states')
            %   OUPUTS: A = State transition matrix
            %   MATLAB functions called: length find size
            %   Non-MATLAB functions called: State_Transition_Matrix_D1
            %   LIMITATIONS: If D>1, n is restricted to 9
            
            if D == 1
                [A] = State_Transition_Matrix_D1(obj,y,n);
            else
                A = zeros(n^D,n^D);
                P = zeros(length(states),1);
                
                for i = D:length(y)-1
                    temp1 = 0;
                    temp2 = 0;
                    for j = D:-1:1
                        temp1 = temp1 + 10^(j-1)*y(i-j+1);
                        temp2 = temp2 + 10^(j-1)*y(i-j+2);
                    end
                    A(find(states==temp1),find(states==temp2)) = A(find(states==temp1),find(states==temp2))+1;
                    P(find(states==temp1)) = P(find(states==temp1)) + 1;
                end
                
                for i = 1:size(A,1)
                    A(i,:) = (A(i,:)+1)/(P(i)+n);
                end
                
                P = (P+1)/(length(states) + sum(P));
            end
        end
        
        function [A] = State_Transition_Matrix_D1(obj,y,n)
            %   Code written by Chandrachur Bhattacharya @ Penn State
            %   Date Last Modified: November 25, 2019
            %   FORMAT: [A] = State_Transition_Matrix_D1(obj,y,n)
            %   PURPOSE: Sub-sub-routine to find the transition matrix when Markov
            %            depth (D) is 1
            %   INPUTS: y = Symbol sequence,
            %           n = Alphabet size,
            %   OUPUTS: A = State transition matrix
            %   MATLAB functions called: None
            %   Non-MATLAB functions called: None
            
            A = zeros(n);
            pre = [1:n];
            
            P = zeros(n,1);
            
            for i = 1:length(y)-1
                A(y(i),y(i+1)) = A(y(i),y(i+1)) + 1;
                P(y(i)) = P(y(i)) + 1;
            end
            
            for i = 1:size(A,1)
                A(i,:) = (A(i,:)+1)/(P(i)+n);
            end
        end
        
        function [y] = symbolize_partitions_unit_range(obj,x,bounds)
            %   Code written by Chandrachur Bhattacharya @ Penn State
            %   Date Last Modified: November 25, 2019
            %   FORMAT: [y] = symbolize_partitions_unit_range(obj,x,bounds)
            %   PURPOSE: This function returns the symbol sequence obtained from
            %            normalizing and partitioning the raw data time-series using
            %            preset partition boundaries
            %   INPUTS: x = Raw time-series,
            %           bounds = Preset partition boundaries
            %   OUPUTS: y = Computed symbol sequence (of same length as 'x')
            %   MATLAB functions called: length
            %   Non-MATLAB functions called: normalizeTS
            
            x = normalize_unit_range(PFSA,x);
            y = zeros(length(x),1);
            
            for i = 1:length(bounds)-1
                y(find(x>=bounds(i) & x<=bounds(i+1))) = i;
            end
            
            y(find(x<=bounds(1))) = 1;
            y(find(x>=bounds(end))) = length(bounds)-1;
        end
        
        function [y] = symbolize_partitions_unit_var(obj,x,bounds)
            %   Code written by Chandrachur Bhattacharya @ Penn State
            %   Date Last Modified: November 25, 2019
            %   FORMAT: [y] = symbolize_partitions_unit_var(obj,x,bounds)
            %   PURPOSE: This function returns the symbol sequence obtained from
            %            normalizing and partitioning the raw data time-series using
            %            preset partition boundaries
            %   INPUTS: x = Raw time-series,
            %           bounds = Preset partition boundaries
            %   OUPUTS: y = Computed symbol sequence (of same length as 'x')
            %   MATLAB functions called: length
            %   Non-MATLAB functions called: normalizeTS
            
            x = normalize_unit_var(obj,x);
            y = zeros(length(x),1);
            
            for i = 1:length(bounds)-1
                y(find(x>=bounds(i) & x<=bounds(i+1))) = i;
            end
            
            y(find(x<=bounds(1))) = 1;
            y(find(x>=bounds(end))) = length(bounds)-1;
        end
        
        function [y] = symbolize_unit_var(obj,x,t,n)
            %   Code written by Chandrachur Bhattacharya @ Penn State
            %   Date Last Modified: November 25, 2019
            %   FORMAT: [y] = symbolize_unit_var(obj,x,t,n)
            %   PURPOSE: This function returns the symbol sequence obtained from
            %            normalizing to unit range of data and partitioning
            %   INPUTS: x = Raw time-series,
            %           t = Type of partitioning schemes (MEP: 'm', Uniform: 'u',
            %               K-means: 'k'),
            %           n = Alphabet size
            %   OUPUTS: y = Computed symbol sequence (of same length as 'x')
            %   MATLAB functions called: length
            %   Non-MATLAB functions called: normalize
            
            x = normalize_unit_var(obj,x);
            y = zeros(length(x),1);
            
            switch t
                
                case 'u'
                    for i = 1:n
                        y(find(x>=(min(x)+(i-1)/n) & x<(min(x)+i/n))) = i;
                    end
                    y(find(y==0)) = n;
                    
                case 'm'
                    x1 = sort(x);
                    K = max(size(x1));
                    lim(1) = min(x1);
                    for i=1:n
                        lim(i+1) = x1(ceil(i*K/n));
                    end
                    for i = 1:n
                        y(find(x>=lim(i) & x<=lim(i+1))) = i;
                    end
                    
                case 'k'
                    y = kmeans(x',n,'MaxIter',10000);
                    
            end
        end
        
        function [y] = symbolize_unit_range(obj,x,t,n)
            %   Code written by Chandrachur Bhattacharya @ Penn State
            %   Date Last Modified: November 25, 2019
            %   FORMAT: [y] = symbolize_unit_range(obj,x,t,n)
            %   PURPOSE: This function returns the symbol sequence obtained from
            %            normalizing to unit range of data and partitioning
            %   INPUTS: x = Raw time-series,
            %           t = Type of partitioning schemes (MEP: 'm', Uniform: 'u',
            %               K-means: 'k'),
            %           n = Alphabet size
            %   OUPUTS: y = Computed symbol sequence (of same length as 'x')
            %   MATLAB functions called: length
            %   Non-MATLAB functions called: normalize
            
            x = normalize_unit_range(PFSA,x);
            y = zeros(length(x),1);
            
            switch t
                
                case 'u'
                    for i = 1:n
                        y(find(x>=(min(x)+(i-1)/n) & x<(min(x)+i/n))) = i;
                    end
                    y(find(y==0)) = n;
                    
                case 'm'
                    x1 = sort(x);
                    K = max(size(x1));
                    lim(1) = min(x1);
                    for i=1:n
                        lim(i+1) = x1(ceil(i*K/n));
                    end
                    for i = 1:n
                        y(find(x>=lim(i) & x<=lim(i+1))) = i;
                    end
                    
                case 'k'
                    y = kmeans(x',n,'MaxIter',10000);
                    
            end
        end
        
        function [states] = generate_states(obj,n,D)
            %   Code written by Chandrachur Bhattacharya @ Penn State
            %   Date Last Modified: November 25, 2019
            %   FORMAT: [states] = generate_states(obj,n,D)
            %   PURPOSE: This function generates unmerged PFSA states in the D-markov
            %            setting
            %   INPUTS: n = Alphabet size,
            %           D = Markov depth
            %   OUPUTS: states = PFSA states (unmerged)
            %   MATLAB functions called: str2double dec2base
            %   Non-MATLAB functions called: None
            %   LIMITATIONS: If D>1, n is restricted to 9
            
            add = 0;
            pre = 0;
            for i = 1:D
                add = add + 10^(i-1);
            end
            for i = 0:n^D-1
                states(i+1,1) = str2double(dec2base(i,n)) + add;
            end
        end
        
        function [P] = State_Probability_Vector(obj,y,n,D,states)
            %   Code written by Chandrachur Bhattacharya @ Penn State
            %   Date Last Modified: january 28, 2020
            %   FORMAT: [P] = State_Probability_Vector(obj,y,n,D,states)
            %   PURPOSE: This function generates the state probability
            %   vector given a partitioned time series and the states
            %   INPUTS: y = Symbol sequence,
            %           n = Alphabet size,
            %           D = Markov depth,
            %           states = List of states (can be generated by function
            %           'generate_states')
            %   OUPUTS: P = State probability vector
            %   MATLAB functions called: find
            %   Non-MATLAB functions called: None
            
            P = zeros(1,length(states));
            
            if D == 1
                for i = 1:length(y)-1
                    P(1,y(i)) = P(1,y(i)) + 1;
                end
            else
                for i = D:length(y)-1
                    temp1 = 0;
                    temp2 = 0;
                    for j = D:-1:1
                        temp1 = temp1 + 10^(j-1)*y(i-j+1);
                        temp2 = temp2 + 10^(j-1)*y(i-j+2);
                    end
                    P(find(states==temp1)) = P(find(states==temp1)) + 1;
                end
            end
            
            P = (P+1)/(length(states) + sum(P));
        end
        
    end
end
