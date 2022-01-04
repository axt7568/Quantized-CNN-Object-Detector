%% ========================================= Pre =============================================================
% Setting the Environment
close all
clear all
clc
% Flags to be set
trainNetwork = 0; % Change this value to 1 if you want to train the network
if(trainNetwork == 0)
    load('dnewlyTrainedDetector.mat'); % Loading a pre-trained network
end
weightsVisualization = 0; % Change this value to 1 if you need to visualize the weights
activationsVisualization = 0; % Change this value to 1 if you need to visualize the weights
activationComparison = 0;%Change this value to 1 if you need to compare activation values
% Note : Comments with the tag DISPLAY are to be modified(uncommented) for execution(if necessary)  
%% ===================================== Groundtruth/Labeled Dataset ==========================================
% Contains 295 images 
% Each image contains one or two labeled instances of a vechicle 
unzip vehicleDatasetImages.zip
data = load('vehicleDatasetGroundTruth.mat');
vehicleDataset = data.vehicleDataset;
%% ========================================== Display Data =====================================================
%*******************DISPLAY****************************
% Display first few rows of the data set.
% vehicleDataset(1:4,:) 
vehicleDataset.imageFilename = fullfile(pwd,vehicleDataset.imageFilename);
originalImage = imread('vehicleImages/image_00001.jpg');
%imshow('vehicleImages/image_00001.jpg');
[rows, columns, numberOfColorChannels] = size(originalImage);
%% ======================================= Split the data set into training and testing dataset ===============================
% Whole Dataset -> 295 images : Training Data : 60% -> 177 + Testing Data : 40% -> 118
rng(0);
shuffledIndices = randperm(height(vehicleDataset));
idx = floor(0.6 * length(shuffledIndices));
trainingDataTbl = vehicleDataset(shuffledIndices(1:idx),:);
testDataTbl = vehicleDataset(shuffledIndices(idx+1:end),:);
%% ================================================= Datastores ==================================================
%  create datastores for loading the image and label data during training and evaluation
% Training Data 
imdsTrain = imageDatastore(trainingDataTbl{:,'imageFilename'});
bldsTrain = boxLabelDatastore(trainingDataTbl(:,'vehicle'));
% Testing Data 
imdsTest = imageDatastore(testDataTbl{:,'imageFilename'});
bldsTest = boxLabelDatastore(testDataTbl(:,'vehicle'));
% Combine image and box label datastores.
trainingData = combine(imdsTrain,bldsTrain);
testData = combine(imdsTest,bldsTest);
%*******************DISPLAY****************************
% Display one of the training images and box labels.
% data = read(trainingData);
% I = data{1};
% bbox = data{2};
% annotatedImage = insertShape(I,'Rectangle',bbox);
% annotatedImage = imresize(annotatedImage,2); 
% figure
% imshow(annotatedImage)
%% ============================================ YOLO Object Detection Network ========================================== 
% Input size specified with computation cost in mind(thus reduced)
% Note that since the images are of size 224 * 399 they need to be
% resized(pre-processing)
inputSize = [224 224 3];
% number of object classes to detect 
numClasses = width(vehicleDataset)-1;
% Pre-processing of Training Data 
trainingDataForEstimation = transform(trainingData,@(data)preprocessData(data,inputSize));
% Anchor Estimation and Association
numAnchors = 7;
%*******************DISPLAY****************************
[anchorBoxes, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors);
% Creating YOLO network 
% To view Network use - analyzeNetwork || Deep Network Designer 
% The network can be constructed manually by using the Deep Network
% Designer
layers = [
    imageInputLayer([224 224 3],"Name","imageinput")
    
    convolution2dLayer([3 3],32,"Name","conv_1","Padding",[1,1,1,1])
    batchNormalizationLayer('Name','BN1')
    leakyReluLayer(0.125,'Name','leaky1')
    
    maxPooling2dLayer([2 2],"Name","maxpool_1","Padding","same","Stride",[2 2])
    
    convolution2dLayer([3 3],64,"Name","conv_2","Padding",[1,1,1,1])
    batchNormalizationLayer('Name','BN2')
    leakyReluLayer(0.125,'Name','leaky2')
    
    maxPooling2dLayer([2 2],"Name","maxpool_2","Padding","same","Stride",[2 2])
    
    convolution2dLayer([3 3],128,"Name","conv_3","Padding",[1,1,1,1])
    batchNormalizationLayer('Name','BN3')
    leakyReluLayer(0.125,'Name','leaky3')
    
    convolution2dLayer([1 1],64,"Name","conv_4","Padding",[0,0,0,0])
    batchNormalizationLayer('Name','BN4')
    leakyReluLayer(0.125,'Name','leaky4')
    
    convolution2dLayer([3 3],128,"Name","conv_5","Padding",[1,1,1,1])
    batchNormalizationLayer('Name','BN5')
    leakyReluLayer(0.125,'Name','leaky5')
    
    maxPooling2dLayer([2 2],"Name","maxpool_3","Padding","same","Stride",[2 2])
    
    convolution2dLayer([3 3],256,"Name","conv_6","Padding",[1,1,1,1])
    batchNormalizationLayer('Name','BN6')
    leakyReluLayer(0.125,'Name','leaky6')
    
    convolution2dLayer([1 1],128,"Name","conv_7","Padding",[0,0,0,0])
    batchNormalizationLayer('Name','BN7')
    leakyReluLayer(0.125,'Name','leaky7')
    
    convolution2dLayer([3 3],256,"Name","conv_8","Padding",[1,1,1,1])
    batchNormalizationLayer('Name','BN8')
    leakyReluLayer(0.125,'Name','leaky8')
    
    maxPooling2dLayer([2 2],"Name","maxpool_4","Padding","same","Stride",[2 2])
    
    convolution2dLayer([3 3],512,"Name","conv_9","Padding",[1,1,1,1])
    batchNormalizationLayer('Name','BN9')
    leakyReluLayer(0.125,'Name','leaky9')
    
    convolution2dLayer([1 1],256,"Name","conv_10","Padding",[0,0,0,0])
    batchNormalizationLayer('Name','BN10')
    leakyReluLayer(0.125,'Name','leaky10')
    
    convolution2dLayer([3 3],512,"Name","conv_11","Padding",[1,1,1,1])
    batchNormalizationLayer('Name','BN11')
    leakyReluLayer(0.125,'Name','leak11')
    
    convolution2dLayer([1 1],256,"Name","conv_12","Padding",[1,1,1,1])
    batchNormalizationLayer('Name','BN12')
    leakyReluLayer(0.125,'Name','leaky12')
    
    convolution2dLayer([3 3],512,"Name","conv_13","Padding",[1,1,1,1])
    batchNormalizationLayer('Name','BN13')
    leakyReluLayer(0.125,'Name','leaky13')
    
    maxPooling2dLayer([2 2],"Name","maxpool_5","Padding","same")
    
    convolution2dLayer([3 3],1024,"Name","conv_14","Padding",[1,1,1,1])
    batchNormalizationLayer('Name','BN14')
    leakyReluLayer(0.125,'Name','leaky14')
    
    convolution2dLayer([1 1],512,"Name","conv_15","Padding",[0,0,0,0])
    batchNormalizationLayer('Name','BN15')
    leakyReluLayer(0.125,'Name','leaky15')
    
    convolution2dLayer([3 3],1024,"Name","conv_16","Padding",[1,1,1,1])
    batchNormalizationLayer('Name','BN16')
    leakyReluLayer(0.125,'Name','leaky16')
    
    convolution2dLayer([1 1],42,"Name","conv_17","Padding",[0,0,0,0])
    batchNormalizationLayer('Name','BN17')
    leakyReluLayer(0.125,'Name','leaky17')
    
    yolov2TransformLayer(7,"Name","yolov2-transform")
    yolov2OutputLayer(anchorBoxes,"Name","yolov2-out")];

lgraph = layerGraph(layers);
%*******************DISPLAY****************************
% To have a look at the layer elements 
analyzeNetwork(lgraph) 
%% =========================================== Data Augmentation =========================================================
% Used to improve Network Accuracy 
% Adds more mods to the trainingData without having to need more labelled
% data. Note : Not done for Test data(due to bias)
augmentedTrainingData = transform(trainingData,@augmentData);
% Visualize the augmented images.
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},'Rectangle',data{2});
    reset(augmentedTrainingData);
end
%*******************DISPLAY****************************
% figure
% montage(augmentedData,'BorderSize',10)
%% =========================================== Pre-processing Training Data ================================================= 
% Pre-process augmented data(convert to size - 224 x 224)
preprocessedTrainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
%*******************DISPLAY****************************
% data = read(preprocessedTrainingData);
% I = data{1};
% bbox = data{2};
% annotatedImage = insertShape(I,'Rectangle',bbox);
% annotatedImage = imresize(annotatedImage,2);
% figure
% imshow(annotatedImage)
%% ========================================== Train YOLO Object detector ==================================================== 
% Training the created network usign Matlab's trainYolov2Objectdetector
% function
loopCount = 0;
converge = 0;
while(converge == 0 && trainNetwork == 1)
    % Training options
    %'LearnRateSchedule','piecewise', ...
    %'LearnRateDropFactor',0.2, ...
    %'LearnRateDropPeriod',40, ...
    options = trainingOptions('sgdm', ...
            'MiniBatchSize',2, ....
            'InitialLearnRate',1e-2, ...
            'MaxEpochs',35, ...
            'Shuffle','never');
%     if(loopCount == 0)
    [detector,info] = trainYOLOv2ObjectDetector(preprocessedTrainingData,lgraph,options);
%         loopCount = loopCount + 1;
%     else 
%         [detector,info] = trainYOLOv2ObjectDetector(preprocessedTrainingData,detector,options);
%     end
    
    save('newlyTrainedDetector.mat','detector') % saving detector
%     
%     if max(info.TrainingRMSE) > 1
%        converge = 0;
%     else
%        % we have converged
%        converge = 1;
%     end
    converge = 1;
end
%% ============================================== Training Accuracy =============================================================
% Displays training accuracy wrt each iteration(if the model was trained)
if(trainNetwork == 1)
    figure
    plot(info.TrainingLoss)
    grid on
    xlabel('Number of Iterations')
    ylabel('Training Loss for Each Iteration')
end
%% ==================================== Weights Size Calculation : Before Quantization ============================================
net = detector.Network; % Setting variable net to poin to the DAG network
layers = [2 6 10 16 20 23 26 30 33 36 39 42 46 49 52 55]; % Conv layer numbers specified as an array
printStatement = sprintf('Layer Weights Size Before Quantization : \n---------------------------------------------\n' );
disp(printStatement)
total = 0; 
for layer = layers
    layerName = net.Layers(layer).Name;  
    layerWeights = net.Layers(layer).Weights;
    layerWeightsinfo = whos('layerWeights');
    layerKB = layerWeightsinfo.bytes/1024; % To calculate the number of KB from bytes
    printStatement = sprintf('%s  -------------------> %f KB',layerName,layerKB);
    disp(printStatement)
    total = total + layerKB;
end
total = total / 1024; % To calculate the number of MB from KB
printStatement = sprintf('Total Size -------------------> %f MB',total); 
disp(printStatement)
%% ================================================ Weights Quantization ==========================================================
% Creating dummy objects
dummyModel  = detector.saveobj; 
tempNetwork = detector.Network.saveobj; 
% warning('off','fixed:numerictype:invalidMinWordLength') % Suppress 0 worldlenght warning
layers = [2 6 10 16 20 23 26 30 33 36 39 42 46 49 52 55]; % Conv layer numbers specified as an array % Conv layer numbers specified as an array
printStatement = sprintf('\nLayer Weights Size after Quantization: \n---------------------------------------------\n');
disp(printStatement)
total = 0; 
for layer = layers
    layerName = tempNetwork.Layers(layer).Name;
    layerWeights = tempNetwork.Layers(layer).Weights;
    % Quantization Algorithms
    %currentWeights = (2.^(nextpow2(layerWeights))).*sign(layerWeights);
    %currentWeights = 2.*floor(layerWeights./2);
    % requiredType = numerictype(1,7,7);
    % currentWeights = quantize(fixedpointWeights,requiredType);
    %currentWeights = fi(layerWeights,1,7,7); % Quantizing the weights to 15-bit Fixed-point
    currentWeights = layerWeights;
    temp = nextpow2(layerWeights) - 2;
    for x = 1:length(layerWeights(:))
      if layerWeights(x) > 0
        currentWeights(x) =  2.^nextpow2(layerWeights(x)-(2.^temp(x)));
      elseif layerWeights(x) < 0 
        currentWeights(x) =  -2.^nextpow2(layerWeights(x)+(2.^temp(x)));  
      end
    end
    % Re-Initializing Weights
    tempNetwork.Layers(layer).Weights = single(currentWeights); 
    layerWeightsinfo = whos('currentWeights');
    layerKB = layerWeightsinfo.bytes/(1024*4);
    printStatement = sprintf('%s  -------------------> %f KB',layerName,layerKB); 
    disp(printStatement)
    total = total + layerKB;
end
total = total /(1024*4); % To calculate the number of MB from KB
printStatement = sprintf('Total Size -------------------> %f MB',total); 
disp(printStatement)
% Conversion of structs to objects to create a new model with quantized
% weights
dummyNetwork = detector.Network.loadobj(tempNetwork);
dummyModel.Network = dummyNetwork;
newModel = detector.loadobj(dummyModel);
%% ============================================= Mean Difference of Weights =======================================================
layers = [2 6 10 16 20 23 26 30 33 36 39 42 46 49 52 55];
totalMean = 0;
count = 0;
for layer = layers
    A = detector.Network.Layers(layer).Weights;
    B = newModel.Network.Layers(layer).Weights;
    meanDiffpercentage = mean(mean((abs(A-B)./abs(A)).*100));
    meaN = mean(meanDiffpercentage(:));
    totalMean = totalMean + meaN;
    count = count + 1;
end
averageMean = totalMean/count;
printStatement = sprintf('\nQuantized Weights Mean Difference -------------------> %f%%\n',averageMean); 
disp(printStatement)
%% ============================================= Accuracy before and after Quantization =======================================================
preprocessedTestData = transform(testData,@(data)preprocessData(data,inputSize));
resultsBeforequantization = detect(detector, preprocessedTestData);
resultsAfterquantization = detect(newModel, preprocessedTestData);
[apBfq, recallBfq, precisionBfq] = evaluateDetectionPrecision(resultsBeforequantization, preprocessedTestData);
[apAfq, recallAfq, precisionAfq] = evaluateDetectionPrecision(resultsAfterquantization, preprocessedTestData);
% Plot recall and precision
figure;
tiledlayout(2,1)
nexttile
plot(recallBfq,precisionBfq);
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average precision = %f Before Quantization', apBfq))
nexttile
plot(recallAfq,precisionAfq);
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average precision = %f After Quantization', apAfq))
%% ================================================= YOLO Detection ================================================================ 
% Running Both the Quantized and original YOLOv2 model for detection on test data 
colorsArray = {'yellow','blue','green','red','black'};
randomIndex = randi(length(testDataTbl.imageFilename),1);
I = imread(testDataTbl.imageFilename{randomIndex});
I = imresize(I,inputSize(1:2));
[bboxes,scores] = detect(detector,I);
[bboxesN,scoresN] = detect(newModel,I);
boxPosition = [bboxes;bboxesN]
boxLabel = [scores;scoresN]
% Yellow -> Original model 
% Blue   -> Quantized model 
if(~isempty(boxPosition) && ~isempty(boxLabel))
    I = insertObjectAnnotation(I,'rectangle',boxPosition,boxLabel,'LineWidth',3,'Color',colorsArray(1:size(boxPosition,1)),'TextColor','black');
    figure
    imshow(I)
else 
    printStatement = sprintf('Please run this section again using the - Run Section button');
    disp(printStatement)
end
%% ================================================ Weights Visualization ==========================================================  
% Single Layer Visualization
if(weightsVisualization == 1)
    layers = [2 6];
    net = detector.Network;
    for layer = layers
        layer_name = net.Layers(layer).Name; % To display the layer name
        channels = 1:32; % This particular conv layer x channels
        I = deepDreamImage(net,layer_name,channels,'PyramidLevels',1);
        figure
        I = imtile(I,'ThumbnailSize',[128 128]);
        imshow(I)
        title(['Layer ',layer_name,' Features'])
    end
end
%% ==================================================== Visualize Activations ======================================================
if(activationsVisualization == 1)
    I = imread(testDataTbl.imageFilename{5});
    I = imresize(I,inputSize(1:2));
    net1 = detector.Network;
    act1 = activations(net1,I,'conv_15');
    sz = size(act1);
    act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
    I = imtile(mat2gray(act1),'GridSize',[4 8]);
    imshow(I)
end