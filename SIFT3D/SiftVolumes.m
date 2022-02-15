%% %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%                                   Preprocessing                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% % Read JSON File with volumes data

fname = 'C:\Users\matte\Desktop\Applied Stat Project\SIFT3D\volumes.json'; 
fid = fopen(fname); 
raw = fread(fid,inf); 
str = char(raw'); 
fclose(fid);
val = jsondecode(str); 
names = fieldnames(val.VOI);
n_volumes = 125; %We have 125 volumes
volumes = cell(n_volumes,1); 


for i = 1:n_volumes
    
    volumes{i} = val.VOI.(string(names(i)));  
    
end

%% % Expand volumes to have more keypoints and to apply SIFT3D

nx=50; ny=50; nz=150; %% desired output dimensions

[x, y, z]=...
    ndgrid(linspace(1,10,ny),...
           linspace(1,10,nx),...
           linspace(1,30,nz));

volumes_expanded = cell(n_volumes,1);

for i = 1:n_volumes
    
    volumes_expanded{i} = interp3(volumes{i},x,y,z,'nearest');
    
end

%% % SIFT3D: Detect keypoints and extract descriptors 

keys = cell(n_volumes,1);
desc = cell(n_volumes,1);
coords = cell(n_volumes,1);
 
for i = 1:n_volumes
    
    keys{i} = detectSift3D(volumes_expanded{i});
    
    [desc{i}, coords{i}] = extractSift3D(keys{i});
    
end

%% %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%                Match the descriptors and compute the matrix of distances            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% % Match the descriptors and compute the matrix of distances 

dist = zeros(n_volumes,n_volumes);

for i = 1:n_volumes
    
    for j = (i+1):n_volumes
        
        matches = matchSift3D(desc{i}, coords{i}, desc{j}, coords{j}, 0.999);
        sz = size(matches);
        desc1 = desc{i};
        desc2 = desc{j};
        
        for k = 1:sz(1)
            
            dist(i,j) = dist(i,j) + norm(desc1(matches(k,1),:)-desc2(matches(k,2),:),2);
            
        end
        
        dist(j,i) = dist(i,j);
        
    end
end

save('DistancesSift.mat','dist'); %Save the distances (useful for clustering for example).

%% %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%                            Supervised classification                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% % Clustering of descriptors and Bag of features 

clear mex

n_centers = 40; %It cannot be more than 40 since one volume has only 40 descriptors
descriptors = zeros(n_volumes*n_centers,768); %768 = length of decriptors in SIFT3D
vocabulary = zeros(n_volumes,n_centers);

for i = 1:n_volumes
    d = desc{i};
    
    [idx, centers] = kmeans(d, n_centers);
    
    counter = zeros(n_centers,1); %counter(i) = number of units in cluster i
    for j = 1:length(idx)
        counter(idx(j)) = counter(idx(j)) + 1;
    end
    vocabulary(i,:) = counter; %Rows of "vocabulary" have the frequencies wrt clusters
    
    descriptors((n_centers*(i-1) + 1) : (n_centers*i),:) = centers;
end

%% % Autoencoder: reduce the length of the descriptors

n_rows = 0;
for i = 1 : n_volumes
    
    sz = size(desc{i});
    n_rows = n_rows + sz(1);
    
end

training_desc = zeros(768, n_rows);
marker = 1;
for i = 1:125
    
    sz = size(desc{i});
    training_desc(:,marker:marker+sz(1)-1) = desc{i}';
    marker = marker + sz(1);    
    
end

hidden_size = 40;
autoenc = trainAutoencoder(training_desc, hidden_size, 'MaxEpoch', 100, 'L2WeightRegularization', 0.05);
output_autoencoder = predict(autoenc,training_desc);
mseError = mse(training_desc-output_autoencoder);

disp("mse Error:"); disp(mseError);

descriptors_autoencoder = encode(autoenc, descriptors');
descriptors_autoencoder = descriptors_autoencoder';


%% % True labels for Classification

dataset = readtable('C:\Users\matte\Desktop\Applied Stat Project\SIFT3D\Database CALI without NAN.xlsx');
patients = string(struct2cell(val.Patient_ID));

%We set the labels as -1 to detect patients without information in the dataset file (NAN)
SOS = -1*ones(n_volumes,1); % 1=presence of SOS vs 0=lack of SOS
FIBROSIS_CENTROLOBULAR = -1*ones(n_volumes,1);
FIBROSIS_PERISINUSOIDAL = -1*ones(n_volumes,1);
PELIOSIS = -1*ones(n_volumes,1);
NRH = -1*ones(n_volumes,1);
STEATOSIS = -1*ones(n_volumes,1);
LOBULAR_FLOGOSIS = -1*ones(n_volumes,1);
BALOONING = -1*ones(n_volumes,1);
STEATOHEPATITIS = -1*ones(n_volumes,1);

CALI = -1*ones(n_volumes,1); %1=presence of CALI vs 0=lack of CALI

threshold = 5;
NUMBER_OF_CALI = -1*ones(n_volumes,1); %1=more than "threshold" CALI vs 0=less than "threshold" CALI

%Read labels from dataset file and save them
for i = 1:length(dataset.Num)
    
    index = find(patients==string(dataset.CODE_ID(i)));
    
    if ~isempty(index)
        
        SOS(index) = 0;
        SOS(index) = 1*(dataset.SOS(i) >= 1);
        
        FIBROSIS_PERISINUSOIDAL(index) = 0;
        FIBROSIS_PERISINUSOIDAL(index) = 1*(dataset.FibrosisPerisinusoidal(i) >= 1);
        
        FIBROSIS_CENTROLOBULAR(index) = 0;
        FIBROSIS_CENTROLOBULAR(index) = 1*(dataset.FibrosisCentrolobular(i) >= 1);
        
        PELIOSIS(index) = 0;
        PELIOSIS(index) = 1*(dataset.Peliosis(i) >= 1);
        
        NRH(index) = 0;
        NRH(index) = 1*(dataset.NRH(i) >= 1);
        
        STEATOSIS(index) = 0;
        STEATOSIS(index) = 1*(dataset.Steatosis(i) >= 2);
        
        LOBULAR_FLOGOSIS(index) = 0;
        LOBULAR_FLOGOSIS(index) = 1*(dataset.LobularFlogosis(i) >= 1);
        
        BALOONING(index) = 0;
        BALOONING(index) = 1*(dataset.Balooning(i) >= 1);
        
        STEATOHEPATITIS(index) = 0;
        STEATOHEPATITIS(index) = 1*(dataset.Steatohepatitis(i) >= 1);
        
        CALI(index) = 0;
        CALI(index) = 1*(SOS(index)==1 | FIBROSIS_PERISINUSOIDAL(index)==1 | FIBROSIS_CENTROLOBULAR(index)==1 | PELIOSIS(index)==1 | NRH(index)==1 | STEATOSIS(index)==1 | LOBULAR_FLOGOSIS(index)==1 | BALOONING(index)==1 | STEATOHEPATITIS(index)==1);
        
        counter_ = (SOS(index)+FIBROSIS_PERISINUSOIDAL(index)+FIBROSIS_CENTROLOBULAR(index)+PELIOSIS(index)+NRH(index)+STEATOSIS(index)+LOBULAR_FLOGOSIS(index)+BALOONING(index)+STEATOHEPATITIS(index));
        
        NUMBER_OF_CALI(index) = 0;
        NUMBER_OF_CALI(index) = 1*(counter_ >= threshold);
        
    end
            
end

%% %Delete the patients without information about their cali in the dataset

count = 0;

for i = 1:125
    
    idx = i-count;
     
    if  (idx < length(SOS)) && (SOS(idx) == -1) %Delete the untouched patients: no data about their cali
        
        SOS(idx) = [];
        FIBROSIS_CENTROLOBULAR(idx) = [];
        FIBROSIS_PERISINUSOIDAL(idx) = [];
        PELIOSIS(idx) = [];
        NRH(idx) = [];
        STEATOSIS(idx) = [];
        LOBULAR_FLOGOSIS(idx) = [];
        BALOONING(idx) = [];
        STEATOHEPATITIS(idx) = [];
        CALI(idx) = [];
        NUMBER_OF_CALI(idx) = []; 
        vocabulary(idx,:) = [];
        descriptors((idx-1)*n_centers+1:idx*n_centers,:) = [];
        descriptors_autoencoder((idx-1)*n_centers+1:idx*n_centers,:) = [];
       
        count = count + 1;
    end
   
    
end

%% % Further targets and true labels combinations

N = length(SOS); %Number of patients with information about their cali

threshold1 = 2;
COMBO_STEATOHEPATITIS_PELIOSIS_SOS_NRH = zeros(N,1); %1=patient with 2 or more cali among Steatohep.,Peliosis,SOS,NRH vs 0=otherwise

threshold2 = 2;
COMBO_STEATOHEPATITIS_PELIOSIS_NRH = zeros(N,1); %1=patient with 2 or more cali among Steatohep.,Peliosis,NRH vs 0=otherwise

threshold3 = 2;
COMBO_STEATOHEPATITIS_PELIOSIS_SOS = zeros(N,1); %1=patient with 2 or more cali among Steatohep.,Peliosis,SOS vs 0=otherwise

threshold4 = 1;
COMBO_STEATOHEPATITIS_PELIOSIS = zeros(N,1); %1=patient with 1 or more cali among Steatohep.,Peliosis vs 0=otherwise

for i = 1:N
    
    counter1 = (SOS(i)+PELIOSIS(i)+NRH(i)+STEATOHEPATITIS(i));
    COMBO_STEATOHEPATITIS_PELIOSIS_SOS_NRH(i) = 1*(counter1 >= threshold1);
    
    counter2 = (PELIOSIS(i)+NRH(i)+STEATOHEPATITIS(i));
    COMBO_STEATOHEPATITIS_PELIOSIS_NRH(i) = 1*(counter2 >= threshold2);
    
    counter3 = (SOS(i)+PELIOSIS(i)+STEATOHEPATITIS(i));
    COMBO_STEATOHEPATITIS_PELIOSIS_SOS(i) = 1*(counter3 >= threshold3);
    
    counter4 = (PELIOSIS(i)+STEATOHEPATITIS(i));
    COMBO_STEATOHEPATITIS_PELIOSIS(i) = 1*(counter4 >= threshold4); 
    
end


%% % SVM and DECISION TREE Classification with "vocabulary"

rng('shuffle')

N = length(SOS);
my_dataset = vocabulary;
target = CALI; %set the true labels to use

target_vs_no_cali = true; % True if you want to classify extreme groups only: 1 --> given by the chosen label; 0 ---> patients with NO cali
if target_vs_no_cali 
    
    new_target = [];
    new_indexes = [];
    count = 1;
    
    for i = 1:N
        
        if (target(i) == 1) || (CALI(i) == 0)
            
            new_target(count) = 1*(target(i) == 1) + 0*(CALI(i) == 0);
            new_indexes(count) = i;
            count = count + 1;
            
        end
        
    end

    N = length(new_target);
    my_dataset = my_dataset(new_indexes,:);
    target = new_target;
        
end


adasyn = true; %set TRUE if you want to apply ADASYN and deal with unbalanced dataset
if adasyn == true
   [new_data, new_labels] = ADASYN(training_set, true_labels, 0.2);
   my_dataset = [my_dataset; new_data];
   target = [target new_labels'];
   N = length(target);
end

n_samples = N; %n_samples used for training: choose N if you want training set only; choose (N-n_testset) if you want a test set.

random_indexes = randperm(N); %random permutation to create random test set

training_set = my_dataset(random_indexes(1:n_samples),:);
true_labels = target(random_indexes(1:n_samples));

if n_samples == N
    test_set = training_set;
    test_true_labels = true_labels;
else
    test_set = my_dataset(random_indexes(n_samples+1:end),:);
    test_true_labels = target(n_samples+1:end);
end


%select the wanted classifiers (SVM, DECISION TREE or ENSEMBLE model)
Model = fitcsvm(training_set,true_labels,'KernelFunction','linear','Standardize',true);
%Model = fitcsvm(training_set,true_labels,'KernelFunction','polynomial','polynomialOrder',2,'Standardize',true);
%Model = fitcsvm(training_set,true_labels,'KernelFunction','rbf','Standardize',true); %'rbf'=radial basis function 
%Model = fitctree(training_set,true_labels);
%Model = fitcensemble(training_set,true_labels,'Method','Bag'); %Alternative:'Method','RUSBoost','LearnRate',0.2,'RatioToSmallest',[1,1]) ---> trained on balanced data in this way (each classifier is trained on a dataset with same number of 0 and 1)

[predicted_labels,~] = predict(Model,test_set);

true_pos = 0;
true_neg = 0;
false_pos = 0;
false_neg = 0;

if n_samples == N
    
    for i = 1:n_samples
        
        if((test_true_labels(i) == 1 && predicted_labels(i) == 1))
            true_pos = true_pos + 1;
        end
        
        if((test_true_labels(i) == 0 && predicted_labels(i) == 0))
            true_neg = true_neg + 1;
        end
        
        if((test_true_labels(i) == 0 && predicted_labels(i) == 1))
            false_pos = false_pos + 1;
        end
        
        if((test_true_labels(i) == 1 && predicted_labels(i) == 0))
            false_neg = false_neg + 1;
        end
        
    end
    
    accuracy = (true_pos + true_neg)/n_samples;

else
    
    for i = 1:(N-n_samples)
        
        if((test_true_labels(i) == 1 && predicted_labels(i) == 1))
            true_pos = true_pos + 1;
        end
        
        if((test_true_labels(i) == 0 && predicted_labels(i) == 0))
            true_neg = true_neg + 1;
        end
        
        if((test_true_labels(i) == 0 && predicted_labels(i) == 1))
            false_pos = false_pos + 1;
        end
        
        if((test_true_labels(i) == 1 && predicted_labels(i) == 0))
            false_neg = false_neg + 1;
        end
        
    end
     
    accuracy = (true_pos + true_neg)/(N-n_samples);
end

disp('Accuracy:'); disp(accuracy);

f_score = true_pos/(true_pos+0.5*(false_pos+false_neg));
disp('F score:'); disp(f_score);

CVModel = crossval(Model,'kFold',5);
classLoss = kfoldLoss(CVModel,'lossfun','classiferror');
disp('kFoldLoss:'); disp(classLoss);

if n_samples ~= N
    
    confusion_matrix = [true_neg, false_pos; false_neg, true_pos];
    disp('Confusion Matrix:'); disp(confusion_matrix);
    
    plotconfusion(categorical(test_true_labels'), categorical(predicted_labels))
    
    disp('--------------------------------------------------------')

end

%% % SVM and DECISION TREE Classification with "descriptors"

rng('shuffle')

N = length(SOS)*n_centers;
my_dataset = descriptors; %select descriptors or descriptors_autoencoder
target = NUMBER_OF_CALI; %set the true labels to use

target_large = reshape(repmat(target,n_centers,1),1,[]);

target_vs_no_cali = true; % True if you want to classify extreme groups only: 1 --> given by the chosen label; 0 ---> patients with no cali
if target_vs_no_cali 
    
    new_target = [];
    new_target1 = [];
    new_indexes = [];
    count = 1;
    
    for i = 1:N
        
        CHECK_CALI = reshape(repmat(CALI,n_centers,1),1,[]);
        
        if (target_large(i) == 1) || (CHECK_CALI(i) == 0)
            
            new_target(count) = 1*(target_large(i) == 1) + 0*(CHECK_CALI(i) == 0);
            new_indexes(count) = i;
            count = count + 1;        
            
        end
        
    end
    
    count=1;    
    for i = 1:length(target)
        
       if (target(i) == 1) || (CALI(i) == 0)
            
            new_target1(count) = 1*(target(i) == 1) + 0*(CALI(i) == 0);
            count = count + 1;        
            
       end
        
    end

    N = length(new_target);
    my_dataset = my_dataset(new_indexes,:);
    target_large = new_target;
    target = new_target1;
    
end


%select the classifier to use (SVM or DECISION TREE)
%Model = fitcsvm(my_dataset,target_large,'KernelFunction','linear','Standardize',true);
%Model = fitcsvm(my_dataset,target_large,'KernelFunction','polynomial','polynomialOrder',2,'Standardize',true);
%Model = fitcsvm(my_dataset,target_large,'KernelFunction','rbf','Standardize',true); %'rbf'=radial basis function 
Model = fitctree(my_dataset,target_large);

[predicted_labels,score] = predict(Model,my_dataset);

%From predicted_labels for descriptors to predicted_labels for each patient
predicted_labels_patients = [];
j = 1;
for i = 0:n_centers:(N-n_centers)
    
    if sum(predicted_labels(i+1:i+n_centers))/n_centers >= 0.5        
        predicted_labels_patients(j) = 1;
        j = j+1;
        
    else 
        predicted_labels_patients(j) = 0;
        j=j+1;
        
    end
       
end
    

true_pos = 0;
true_neg = 0;
false_pos = 0;
false_neg = 0;

for i = 1:length(target)
    
    if((target(i) == 1 && predicted_labels_patients(i) == 1))
            true_pos = true_pos + 1;
    end
        
    if((target(i) == 0 && predicted_labels_patients(i) == 0))
            true_neg = true_neg + 1;
    end
        
    if((target(i) == 0 && predicted_labels_patients(i) == 1))
            false_pos = false_pos + 1;
    end
        
    if((target(i) == 1 && predicted_labels_patients(i) == 0))
            false_neg = false_neg + 1;
    end
        
end
 
accuracy = (true_pos + true_neg)/length(target);
disp('Accuracy:'); disp(accuracy);

f_score = true_pos/(true_pos+0.5*(false_pos+false_neg));
disp('F score:'); disp(f_score);

disp('Target:'); disp(target');
disp('Predicted Labels:'); disp(predicted_labels_patients');