classdef PFSA
    %   Class File Containing the Essential PFSA Functions needed
    %   Code written by Chandrachur Bhattacharya @ Penn State
    %   Date Last Modified: August, 2021
    %   
    %   Please cite:
    %   Chandrachur Bhattacharya and Dr Asok Ray. Thresholdless Classification
    %   of Chaotic Dynamics and Combustion Instability via Probabilistic Finite
    %   State Automata." Mechanical Systems and Signal Processing (2021).
    %
    %   Important functions in this class and their syntax given below:
    %   (For details please type: help PFSA/function_name)
    %
    %   FUNCTIONS IMPORTANT TO THE USER
    %
    %   [model] = Define_PFSA(N,D,part_scheme,global_part,unit_var,analysis_type)
    %   [model] = Train_PFSA_Bulk(model,training_data,training_label)
    %   [model] = Init_PFSA_Sequential(model,training_data_raw)
    %   [model] = Train_PFSA_Sequential(model,training_data,training_label)
    %   [pred_class] = Test_PFSA(model,testing_data)
    %
    %   There are 2 routes to training and testing PFSA models
    %
    %   ROUTE 1:
    %   Define_PFSA -> Train_PFSA_Bulk -> Test_PFSA
    %
    %   (Preferred when training data windows are lower in number / data
    %   pre-organized as [TS_data_window_1; ... ;TS_data_window_n] and
    %   [label_1; ... ; label_n] format)
    %
    %   ROUTE 2:
    %   Define_PFSA -> Init_PFSA_Sequential -> Train_PFSA_Sequential -> Test_PFSA
    %
    %   (Preferred when training data windows are large in number / data
    %   pre-organized as [TS_data_1; ... ;TS_data_n] and
    %   [label_series_1; ... ; label_series_n] format)
    %
    %   NOTE: Sequential training is sometimes a bit slower than bulk
    %   training, due to the fact that values are recomputed at every step.
    %   However, this allows for lesser data manipulation (like needed in
    %   the 'Bulk' method). This leads to an overall faster training.
    %
    %   FOR A LISTING OF THE OTHER FUNCTIONS IN THIS CLASS PLEASE TYPE:
    %   help PFSA/More_Info
    
    methods(Static)
        function [model] = Define_PFSA(N,D,part_scheme,global_part,unit_var,analysis_type)
            %   Code written by Chandrachur Bhattacharya @ Penn State
            %   Date Last Modified: May 6, 2020
            %
            %   FORMAT: [model] = Define_PFSA(training_data,training_label,...
            %       N,D,partitioning,global_part,unit_var,analysis_type)
            %   PURPOSE: Creates the PFSA model to be used for the analysis
            %            Outputs a PFSA model that is used during training 
            %            and testing phases.
            %            [Needs to be called as an object]
            %
            %   INPUTS: N =             Number of partitions of the PFSA
            %           D =             Depth of the PFSA
            %           global_part =   Set to 1 if a global partition is
            %                           desired (recommended), 0 otherwise
            %           unit_var =      Set to 1 if normalization is to be 
            %                           done to generate zero mean and unit
            %                           variance (recommended), 0 otherwise
            %           analysis_type = Choose the PFSA analysis type:
            %                           's' - Standard PFSA
            %                           'p' - Projection PFSA (recommended)
            %                           'b' - Best (tries to choose the
            %                           most accurate of 's' and 'p')
            %   OUTPUTS: model        = Base PFSA model
            
            model.N = N;
            model.D = D;
            model.part_scheme = part_scheme;
            model.global_part = global_part;
            model.unit_var = unit_var;
            model.analysis_type = analysis_type;
            model.classes = 0;
        end
     
        function [model] = Train_PFSA_Bulk(model,training_data,training_label)
            %   Code written by Chandrachur Bhattacharya @ Penn State
            %   Date Last Modified: May 6, 2020
            %
            %   FORMAT: [model] = Train_PFSA_Bulk(model,training_data,training_label)
            %   PURPOSE: Trains the PFSA model using the training time 
            %            series data and the corresponding (numeric) class
            %            label. 'Bulk' implies full training data and labels
            %            must be provided to the function.
            %            [Needs to be called as an object]
            %            
            %   INPUTS: model         = Base PFSA model with parameters.
            %                           This must be generated using the
            %                           function: Define_PFSA(...)
            %           training_data = Training data time-series as a
            %                           matrix of size (m x n), n is the
            %                           number of samples per time series
            %                           and m is the number of training 
            %                           time series
            %                           [NOTE: All the time series must 
            %                           have the same length (n)]
            %           training_label = Corresponding training labels for 
            %                           the time series as a vector of size 
            %                           (m x 1)
            %   OUTPUTS: model        = Trained PFSA model
            
            N = model.N;
            D = model.D;
            part_scheme = model.part_scheme;
            global_part = model.global_part;
            unit_var = model.unit_var;
            analysis_type = model.analysis_type;
            
            num_classes = length(unique(training_label));
            states = generate_states(PFSA,N,D);
            
            for i = 1:num_classes
                pi_train{i} = zeros(N^D,N^D);
                mean_train(i) = 0;
                std_train(i) = 0;
                cnt(i) = 0;
            end
            
            if global_part == 1
                if unit_var == 1
                    for i = 1:size(training_data,1)
                        bounds(i,:) = return_partitions_unit_var(PFSA,training_data(i,:),part_scheme,N);
                    end
                elseif unit_var == 2
                     for i = 1:size(training_data,1)
                        bounds(i,:) = return_partitions(PFSA,training_data(i,:),part_scheme,N);
                    end
                else                   
                    for i = 1:size(training_data,1)
                        bounds(i,:) = return_partitions_unit_range(PFSA,training_data(i,:),part_scheme,N);
                    end
                end
%                 bounds = sign(mean(bounds)).*max(abs(bounds));
                bounds = sign(mean(bounds)).*mean(abs(bounds));
%                 bounds = mean(bounds);
            end
            
            for i = 1:size(training_data,1)
                if global_part == 1
                    if unit_var == 1
                        y = symbolize_partitions_unit_var(PFSA,training_data(i,:),bounds);
                    elseif unit_var ==2
                        y = symbolize_partitions(PFSA,training_data(i,:),bounds);
                    else
                        y = symbolize_partitions_unit_range(PFSA,training_data(i,:),bounds);
                    end
                else
                    if unit_var == 1
                        y = symbolize_unit_var(PFSA,training_data(i,:),part_scheme,N);
                    else
                        y = symbolize_unit_range(PFSA,training_data(i,:),part_scheme,N);
                    end
                end
                
                pi_temp = State_Transition_Matrix(PFSA,y,N,D,states);
                pi_train{training_label(i)} = pi_train{training_label(i)} + pi_temp;
                
                mean_train(training_label(i)) = mean_train(training_label(i)) + mean(training_data(i,:));
                std_train(training_label(i)) = std_train(training_label(i)) + std(training_data(i,:));
                
                cnt(training_label(i)) = cnt(training_label(i)) + 1;
            end
            
            for i = 1:num_classes
                pi_train{i} = pi_train{i}/cnt(i);
                mean_train(i) = mean_train(i)/cnt(i);
                std_train(i) = std_train(i)/cnt(i);
                
                if analysis_type == 'p' || analysis_type == 'b'
                    [Right{i},Val{i},Left{i}] = eig(pi_train{i},'vector');
                    [~,ind] = sort(real(Val{i}),'descend');
                    Val{i} = Val{i}(ind);
                    
                    Right{i} = Right{i}(:,ind);
                    mult = sign(Right{i}(1,1));
                    Right{i} = Right{i}*mult;
                    
                    Left{i} = Left{i}(:,ind);
                    Left{i} = Left{i}'*mult;
                    
                    factor = sqrt(abs(Val{i}));
                    
                    temp = factor'.*Right{i};
                    
                    for j = 1:1
                        [Right_Ortho{i,j},~] = qr(temp(:,[[1:j-1],[j+1:end]]));
                        Right_Ortho{i,j} = sign(real(Right_Ortho{i,j}(1,[1:end-1]))).*Right_Ortho{i,j}(:,[1:end-1]);
                    end
                end
            end

            model.pi_train = pi_train;
            model.mean_train = mean_train;
            model.std_train = std_train;
            if analysis_type == 'p' || analysis_type == 'b'
                model.Right_Ortho = Right_Ortho;
            end
            if global_part == 1
                model.bounds = bounds;
            end
            
            model.classes = num_classes;
            model.map = [1:num_classes];
        end
        
        function [model] = Init_PFSA_Sequential(model,training_data_raw)
            %   Code written by Chandrachur Bhattacharya @ Penn State
            %   Date Last Modified: May 6, 2020
            %
            %   FORMAT: [Init_PFSA_Sequential(model,training_data_raw)
            %   PURPOSE: Initializes the PFSA model using the training raw
            %            time series data. Here the time-series data need
            %            not be organized and is inputed as a (p x q)
            %            matrix, where p is the number of observed raw time
            %            series (irrespective of regime / class) and q is
            %            the data samples per time series
            %            [Needs to be called as an object]
            %            
            %   INPUTS: model         = Base PFSA model with parameters.
            %                           This must be generated using the
            %                           function: Define_PFSA(...)
            %           training_data_raw = Training data time-series as a
            %                           matrix of size (p x q), p is the 
            %                           number of observed raw time series 
            %                           (irrespective of regime / class) 
            %                           and q is the data samples per time
            %                           series
            %                           [NOTE: All the time series must 
            %                           have the same length (q)]
            %   OUTPUTS: model        = Initialized PFSA model for
            %                           Sequential method
            
            N = model.N;
            D = model.D;
            part_scheme = model.part_scheme;
            global_part = model.global_part;
            unit_var = model.unit_var;
            
            states = generate_states(PFSA,N,D);
            
%             training_data_raw = training_data_raw(:);
            
            if global_part == 1
                if unit_var == 1
                    for i = 1:size(training_data_raw,1)
                        bounds(i,:) = return_partitions_unit_var(PFSA,training_data_raw(i,:),part_scheme,N);
                    end
                elseif unit_var == 2
                    for i = 1:size(training_data_raw,1)
                        bounds(i,:) = return_partitions(PFSA,training_data_raw(i,:),part_scheme,N);
                    end
                else
                    for i = 1:size(training_data_raw,1)
                        bounds(i,:) = return_partitions_unit_range(PFSA,training_data_raw(i,:),part_scheme,N);
                    end
                end
                model.bounds_list = bounds;
        
                bounds = sign(mean(bounds)).*max(abs(bounds));
%                 bounds = mean(bounds);

                if unit_var == 2 || unit_var == 0
                    bounds = [0:1/(length(bounds)-1):1]*(max(bounds) - min(bounds)) + min(bounds);
                end
            end
            
            if global_part == 1
                model.bounds = bounds;
            end
            
        end
       
        function [model] = Train_PFSA_Sequential(model,training_data,training_label)
            %   Code written by Chandrachur Bhattacharya @ Penn State
            %   Date Last Modified: May 6, 2020
            %
            %   FORMAT: [model] = Train_PFSA_Sequential(model,training_data,training_label)
            %   PURPOSE: Trains the PFSA model using the training time 
            %            series data and the corresponding (numeric) class
            %            label. 'Sequential' implies the training data and 
            %            labels is provided to the function sequqnetially 
            %            (i.e. in an online windowed fashion).
            %            [Needs to be called as an object]
            %            
            %   INPUTS: model         = Base PFSA model with parameters.
            %                           This must be generated using the
            %                           function: Define_PFSA(...) and
            %                           initialized by: Init_PFSA_Sequential(...)
            %           training_data = Training data time-series as a
            %                           vector of size (1 x n), n is the
            %                           number of samples in the time series
            %                           [NOTE: Only a single time series is
            %                           allowed]
            %           training_label = Corresponding training label for 
            %                           the time series as a scalar
            %   OUTPUTS: model        = Trained PFSA model
            
            N = model.N;
            D = model.D;
            part_scheme = model.part_scheme;
            global_part = model.global_part;
            unit_var = model.unit_var;
            analysis_type = model.analysis_type;
            if global_part == 1
                bounds = model.bounds;
            end
     
            states = generate_states(PFSA,N,D);
            
            if model.classes == 0
                num_classes = model.classes;
                
                num_classes = num_classes + 1;
                pi_train{1} = zeros(N^D,N^D);
                mean_train(1) = 0;
                std_train(1) = 0;
                cnt(1) = 0;
                map(1) = training_label;
            elseif ismember(training_label,model.map) == 0
                num_classes = model.classes;
                pi_train = model.pi_train;
                mean_train = model.mean_train;
                std_train = model.std_train;
                cnt = model.cnt;
                map = model.map;
                if analysis_type == 'p' || analysis_type == 'b'
                    Right_Ortho = model.Right_Ortho;
                end
                
                num_classes = num_classes + 1;
                pi_train{end+1} = zeros(N^D,N^D);
                mean_train(end+1) = 0;
                std_train(end+1) = 0;
                cnt(end+1) = 0;
                map(end+1) = training_label;
                
            else
                num_classes = model.classes;
                pi_train = model.pi_train;
                mean_train = model.mean_train;
                std_train = model.std_train;
                cnt = model.cnt;
                map = model.map;
                if analysis_type == 'p' || analysis_type == 'b'
                    Right_Ortho = model.Right_Ortho;
                end
            end
 
            if global_part == 1
                if unit_var == 1
                    y = symbolize_partitions_unit_var(PFSA,training_data(1,:),bounds);
                elseif unit_var ==2
                    y = symbolize_partitions(PFSA,training_data(1,:),bounds);
                else
                    y = symbolize_partitions_unit_range(PFSA,training_data(1,:),bounds);
                end
            else
                if unit_var == 1
                    y = symbolize_unit_var(PFSA,training_data(1,:),part_scheme,N);
                else
                    y = symbolize_unit_range(PFSA,training_data(1,:),part_scheme,N);
                end
            end
                
            pi_temp = State_Transition_Matrix(PFSA,y,N,D,states);
            pi_train{training_label(1)} = (pi_train{training_label(1)}*cnt(training_label(1))...
                + pi_temp)/(cnt(training_label(1)) + 1);
            
            mean_train(training_label(1)) = (mean_train(training_label(1))*cnt(training_label(1))...
                + mean(training_data(1,:)))/(cnt(training_label(1)) + 1);
            std_train(training_label(1)) = (std_train(training_label(1))*cnt(training_label(1))...
                + std(training_data(1,:)))/(cnt(training_label(1)) + 1);
            
            cnt(training_label(1)) = cnt(training_label(1)) + 1;
            
            if analysis_type == 'p' || analysis_type == 'b'
                [Right,Val,Left] = eig(pi_train{training_label(1)},'vector');
                [~,ind] = sort(real(Val),'descend');
                Val = Val(ind);

                Right = Right(:,ind);
                mult = sign(Right(1,1));
                Right = Right*mult;

                Left = Left(:,ind);
                Left = Left'*mult;

                factor = sqrt(abs(Val));

                temp = factor'.*Right;

                for j = 1:1
                    [Right_Ortho{training_label(1),j},~] = qr(temp(:,[[1:j-1],[j+1:end]]));
                    Right_Ortho{training_label(1),j} = sign(real(Right_Ortho{training_label(1),j}(1,[1:end-1])))...
                        .*Right_Ortho{training_label(1),j}(:,[1:end-1]);
                end
                
            end

            model.pi_train = pi_train;
            model.mean_train = mean_train;
            model.std_train = std_train;
            model.cnt = cnt;
            if analysis_type == 'p' || analysis_type == 'b'
                model.Right_Ortho = Right_Ortho;
            end
            if global_part == 1
                model.bounds = bounds;
            end
            
            model.classes = num_classes;
            model.map = map;
        end
                
        function [pred_class] = Test_PFSA(model,testing_data)
            %   Code written by Chandrachur Bhattacharya @ Penn State
            %   Date Last Modified: May 6, 2020
            %
            %   FORMAT: [pred_class] = Test_PFSA(model,testing_data)
            %   PURPOSE: Classifies the test time series data using the
            %            trained PFSA model. The output is the (numeric) 
            %            class label.
            %            [Needs to be called as an object]
            %            
            %   INPUTS: model         = Trained PFSA model with parameters.
            %                           This must be generated using the
            %                           function: Train_PFSA(...)
            %                           (which needs: Define_PFSA(...))
            %           testing_data  = Testing data time-series as a
            %                           matrix of size (m x n), n is the
            %                           number of samples per time series
            %                           and m is the number of training 
            %                           time series. 
            %                           [NOTE 1: All the time series must 
            %                           have the same length (n)]
            %                           [NOTE 2: To analyse a single window 
            %                           (e.g. online) the matrix is
            %                           a vector of size (1 x n)]
            %   OUTPUTS: pred_class =   Corresponding predicted classification
            %                           labels for the test time series
            %                           as a vector of size (m x 1).
            %                           [NOTE: When analysing a single window 
            %                           (e.g. online) output is a scalar]
            
            N = model.N;
            D = model.D;
            part_scheme = model.part_scheme;
            global_part = model.global_part;
            unit_var = model.unit_var;
            analysis_type = model.analysis_type;
            
            pi_train = model.pi_train;
            mean_train = model.mean_train;
            std_train = model.std_train;
            if analysis_type == 'p' || analysis_type == 'b'
                Right_Ortho = model.Right_Ortho;
            end
            if global_part == 1
                bounds = model.bounds;
            end
            map = model.map;
            num_classes = model.classes;
            
            states = generate_states(PFSA,N,D);
            
            for i = 1:size(testing_data,1)
                if global_part == 1
                    if unit_var == 1
                        y = symbolize_partitions_unit_var(PFSA,training_data(i,:),bounds);
                    elseif unit_var ==2
                        y = symbolize_partitions(PFSA,training_data(i,:),bounds);
                    else
                        y = symbolize_partitions_unit_range(PFSA,training_data(i,:),bounds);
                    end
                else
                    if unit_var == 1
                        y = symbolize_unit_var(PFSA,testing_data(i,:),part_scheme,N);
                    else
                        y = symbolize_unit_range(PFSA,testing_data(i,:),part_scheme,N);
                    end
                end
                
                pi_temp = State_Transition_Matrix(PFSA,y,N,D,states);
                std_temp = std(testing_data(i,:));
                mean_temp = mean(testing_data(i,:));
                
                dist_pi = [];
                if analysis_type == 's' || analysis_type == 'b'
                    for j = 1:num_classes
                        dist_pi(j) = norm([norm(pi_train{j} - pi_temp,2),...
                            std_temp-std_train(j),mean_temp - mean_train(j)],2);
%                         dist_pi(j) = norm([norm(pi_train{j} - pi_temp,2),...
%                             (std_temp-std_train(j))/std_train(j),(mean_temp - mean_train(j))/mean_train(j)],2);
                    end
                end
                
                error = [];
                if analysis_type == 'p' || analysis_type == 'b'
                    [Right_temp,Val_temp,Left_temp] = eig(pi_temp,'vector');
                    [~,ind] = sort(real(Val_temp),'descend');
                    Val_temp = Val_temp(ind);
                    Right_temp = Right_temp(:,ind);
                    mult = sign(real(Right_temp(1,1)));
                    Left_temp = (Left_temp(:,ind)*mult)';
                    
                    for j = 1:num_classes
                        error(j) = norm([vecnorm((Left_temp(1,:)*Right_Ortho{j,1}),2,2),...
                            std_temp-std_train(j),mean_temp - mean_train(j)],2);
%                         error(j) = norm([vecnorm((Left_temp(1,:)*Right_Ortho{j,1}),2,2),...
%                             1-min(std_temp/std_train(j),std_train(j)/std_temp),...
%                             1-min(mean_temp/mean_train(j),mean_train(j)/mean_temp)],2);
                    end

%                     for j = 1:num_classes
%                         error(j,1) = vecnorm((Left_temp(1,:)*Right_Ortho{j,1}),2,2);
%                         error(j,2) = std_temp-std_train(j);
%                         error(j,3) = mean_temp - mean_train(j);
%                     end
%                     
%                     [~,p] = sort(abs(error),'ascend');
%                     
%                     r = [];
%                     for k = 1:3
%                         r(p(:,k),k) = [1:num_classes];
%                     end
%                     
%                     error = sum(r')';
                end
                
                switch analysis_type
                    case 's'
                        pred_class(i,1) = map(find(dist_pi == min(dist_pi)));
                    case 'p'
%                         [M,F] = mode(r');
%                         temp_pred = union(find(error == min(error)),find(M == min(M)));
%                         if length(temp_pred) > 1
%                             [M,F] = mode(r(temp_pred,:)');
%                             temp_pred = temp_pred(find(M == min(M)));
%                         end
%                         if length(temp_pred) > 1
%                             for j = 1:num_classes
%                                 error(j) = norm([vecnorm((Left_temp(1,:)*Right_Ortho{j,1}),2,2),...
%                                     std_temp-std_train(j),mean_temp - mean_train(j)],2);
%                             end
%                             temp_pred = find(error == min(error));
%                         end                            
%                             
%                         pred_class(i,1) = map(temp_pred);
  
                        pred_class(i,1) = map(find(error == min(error)));
                    case 'b'
                        conf_pi = sort((1./dist_pi)./sum(1./dist_pi),'descend');
                        conf_proj = sort((1./error)./sum(1./error),'descend');

                        if conf_pi(1) >= conf_proj(1)
                            pred_class(i,1) = map(find(dist_pi == min(dist_pi)));
                        else
                            pred_class(i,1) = map(find(error == min(error)));
                        end
                end
            end
        end
     end
 
     methods
        function [] = More_Info()
            %   OTHER PROGRAM FUNCTIONS
            %   For further information on these functions please type:
            %   "help PFSA/function_name"
            %
            %   [arr] = chooseNfromP(obj,N,P)
            %   [bounds] = return_partitions_unit_var(obj,x,t,n)
            %   [bounds] = return_partitions_unit_range(obj,x,t,n)
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
        end 
         
        function [arr] = chooseNfromP(obj,N,P)
            %   Code written by Chandrachur Bhattacharya @ Penn State
            %   Date Last Modified: November 25, 2019
            %   FORMAT: [arr] = chooseNfromP(obj,N,P)
            %   PURPOSE: This function chooses N numbers from P numbers randomly
            %   INPUTS: N = No. of Numbers to be chosen,
            %           P = Total Numbers
            %   OUTPUTS: arr = Array of N selected numbers
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
            %   OUTPUTS: y =  Normalized time-series
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
            %   OUTPUTS: y =  Normalized time-series
            %   MATLAB functions called: max min
            %   Non-MATLAB functions called: None
            range = max(x) - min(x) + 1e-9;
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
            %   OUTPUTS: bounds = normalized partition boundary values
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
            %   OUTPUTS: bounds = normalized partition boundary values
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

        function [bounds] = return_partitions(obj,x,t,n)
            %   Code written by Chandrachur Bhattacharya @ Penn State
            %   Date Last Modified:August 4, 2021
            %   FORMAT: [bounds] = return_partitions(obj,x,t,n)
            %   PURPOSE: This function returns the partition boundaries for a given raw
            %            data time series without normalizing the data
            %   INPUTS: x = Raw time-series,
            %           t = Type of partitioning schemes (MEP: 'm', Uniform: 'u',
            %               K-means: 'k'),
            %           n = Alphabet size
            %   OUTPUTS: bounds = partition boundary values
            %   MATLAB functions called: min max sort size kmeans find
            %   Non-MATLAB functions called: normalizeTS
            
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
            %   OUTPUTS: A = State transition matrix
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
            %   OUTPUTS: A = State transition matrix
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
            %   OUTPUTS: y = Computed symbol sequence (of same length as 'x')
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
            %   OUTPUTS: y = Computed symbol sequence (of same length as 'x')
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

        function [y] = symbolize_partitions(obj,x,bounds)
            %   Code written by Chandrachur Bhattacharya @ Penn State
            %   Date Last Modified: November 25, 2019
            %   FORMAT: [y] = symbolize_partitions_unit_var(obj,x,bounds)
            %   PURPOSE: This function returns the symbol sequence obtained from
            %            normalizing and partitioning the raw data time-series using
            %            preset partition boundaries
            %   INPUTS: x = Raw time-series,
            %           bounds = Preset partition boundaries
            %   OUTPUTS: y = Computed symbol sequence (of same length as 'x')
            %   MATLAB functions called: length
            %   Non-MATLAB functions called: normalizeTS

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
            %   OUTPUTS: y = Computed symbol sequence (of same length as 'x')
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
            %   OUTPUTS: y = Computed symbol sequence (of same length as 'x')
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
            %   OUTPUTS: states = PFSA states (unmerged)
            %   MATLAB functions called: str2double dec2base
            %   Non-MATLAB functions called: None
            %   LIMITATIONS: If D>1, n is restricted to 9
            
            add = 0;
            pre = 0;
            if D == 1
                states = [1:n];
            else
                for i = 1:D
                    add = add + 10^(i-1);
                end
                for i = 0:n^D-1
                    states(i+1,1) = str2double(dec2base(i,n)) + add;
                end
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
            %   OUTPUTS: P = State probability vector
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
