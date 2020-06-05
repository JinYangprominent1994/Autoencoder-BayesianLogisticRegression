initial_x = readPoints('dat/107_0764.pts');

x_mean = initial_x;

allFiles = dir('dat/107*.pts');

total = zeros(size(initial_x));

N = length(allFiles);

X = zeros(136,21);
    
  for iI =1:length(allFiles)
        
      cPts = readPoints( strcat('dat/',allFiles(iI).name ) );
 
      [ptsA,pars] = getAlignedPts( x_mean, cPts );
      
      singleFace = reshape(ptsA,[136,1])
      
      X(:,iI) = singleFace;
      
      figure(1);drawFaceParts(-ptsA,'k-');
        
  end

% AE with the full size of hidden layers
hiddenSize = 136;

% AE with 1% of the full size of hidden layers
%hiddenSize = 1;

% AE with 3% of the full size of hidden layers
%hiddenSize = 4;

% AE with 10% of the full size of hidden layers
%hiddenSize = 14;

autoenc = trainAutoencoder(X,hiddenSize);

xReconstructed = predict(autoenc,X);

mseError = mse(X-xReconstructed);

  for j =1:length(allFiles)
        
      cPts = readPoints( strcat('dat/',allFiles(j).name ) );
 
      [ptsA,pars] = getAlignedPts( x_mean, cPts );
      
      singleNewData = xReconstructed(:,j);
      
      newData = reshape(singleNewData,[68,2])
      
      figure(2);drawFaceParts( -newData, 'k-' );
        
  end
  
 


 
 