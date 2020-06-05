initial_x = readPoints('dat/107_0764.pts');

x_mean = initial_x;

allFiles = dir('dat/107*.pts');

total = zeros(size(initial_x));

%energy = zeros(1,10);

N = length(allFiles);

%total_energy = 0;

X = zeros(136,21);
    
  for iI =1:length(allFiles)
        
      cPts = readPoints( strcat('dat/',allFiles(iI).name ) );
 
      [ptsA,pars] = getAlignedPts( x_mean, cPts );
      
      singleFace = reshape(ptsA,[136,1])
      
      X(:,iI) = singleFace;
      
      figure(1);drawFaceParts(-ptsA,'k-');
        
  end

hiddenSize = 136;

% AE with regularity with weights 0.1
autoenc = trainAutoencoder(X,hiddenSize,'L2WeightRegularization',0.1);

% AE with regularity with weights 0.2
%autoenc = trainAutoencoder(X,hiddenSize,'L2WeightRegularization',0.2);

% AE with regularity with weights 0.3
%autoenc = trainAutoencoder(X,hiddenSize,'L2WeightRegularization',0.3);

% AE with regularity with weights 0.4
%autoenc = trainAutoencoder(X,hiddenSize,'L2WeightRegularization',0.4);

% AE with regularity with weights 0.5
%autoenc = trainAutoencoder(X,hiddenSize,'L2WeightRegularization',0.5);

xReconstructed = predict(autoenc,X);

mseError = mse(X-xReconstructed);

  for j =1:length(allFiles)
        
      cPts = readPoints( strcat('dat/',allFiles(j).name ) );
 
      [ptsA,pars] = getAlignedPts( x_mean, cPts );
      
      singleNewData = xReconstructed(:,j);
      
      newData = reshape(singleNewData,[68,2])
      
      figure(2);drawFaceParts( -newData, 'k-' );
        
  end