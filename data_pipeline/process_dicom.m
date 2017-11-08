function process_dicom(rootPath, pathToTrainImages, pathToTrainCSV, pathToTestImages, pathToTestCSV, writeDirectory)
    trainDataTable = readtable(strcat(rootPath, pathToTrainCSV));
    tablesize = size(trainDataTable);
    numRows = tablesize(1);
    radius = 375; % cropping to 750 x 750 pixels

    for i = 1:numRows
        dcmMaskFileNameRaw = trainDataTable{i,'ROIMaskFilePath'};
        dcmImgFileNameRaw = trainDataTable{i,'imageFilePath'};
        dcmMaskFileName = char(dcmMaskFileNameRaw);
        dcmImgFileName = char(dcmImgFileNameRaw);
        splitName = strsplit(dcmImgFileName, '/');
        base_name = splitName{1};
        base_name = strrep(base_name,'Mass-Training_','');

        grayImg = dicomread(dicominfo(strcat(strcat(rootPath, pathToTrainImages), dcmMaskFileName)));
        dataType = isa(grayImg(1,1),'uint16');
        if dataType
            dcmMaskFileNameRaw = trainDataTable{i,'croppedImageFilePath'};
            grayImg = dicomread(dicominfo(strcat(strcat(rootPath, pathToTrainImages), char(dcmMaskFileNameRaw))));
        end
        level = graythresh(grayImg);
        BW = imbinarize(grayImg,level);
        I = dicomread(dicominfo(strcat(strcat(rootPath, pathToTrainImages), dcmImgFileName)));
        s = regionprops(BW,'centroid');
        centroids = cat(1, s.Centroid);
        mean_centroid = mean(centroids, 1);

        xLeft = mean_centroid(1) - radius;
        yTop = mean_centroid(2) - radius;
        BW = imcrop(BW, [xLeft, yTop, 749, 749]);
        I = imcrop(I, [xLeft, yTop, 749, 749]);
        I = double(I);
        I = I-mean(I(:));
        I = I/std(I(:),0,1);
        maskWriteRoot = 'train_masks_cropped/';
        imgWriteRoot = 'train_mammo_cropped/';
        mkdir(writeDirectory, 'train_masks_cropped');
        mkdir(writeDirectory, 'train_mammo_cropped');
        imwrite(BW, strcat(writeDirectory, strcat(maskWriteRoot, strcat(base_name, '.tif'))));
        imwrite(I, strcat(writeDirectory, strcat(imgWriteRoot, strcat(base_name, '.tif'))));
    end

    testDataTable = readtable(strcat(rootPath, pathToTestCSV));
    tablesize = size(testDataTable);
    numRows = tablesize(1);

    for i = 1:numRows
        dcmMaskFileNameRaw = testDataTable{i,'ROIMaskFilePath'};
        dcmImgFileNameRaw = testDataTable{i,'imageFilePath'};
        dcmMaskFileName = char(dcmMaskFileNameRaw);
        dcmImgFileName = char(dcmImgFileNameRaw);
        splitName = strsplit(dcmImgFileName, '/');
        base_name = splitName{1};
        base_name = strrep(base_name,'Mass-Test_','');

        grayImg = dicomread(dicominfo(strcat(strcat(rootPath, pathToTestImages), dcmMaskFileName)));
        dataType = isa(grayImg(1,1),'uint16');
        if dataType
            dcmMaskFileNameRaw = testDataTable{i,'croppedImageFilePath'};
            grayImg = dicomread(dicominfo(strcat(strcat(rootPath, pathToTestImages), char(dcmMaskFileNameRaw))));
        end
        level = graythresh(grayImg);
        BW = imbinarize(grayImg,level);
        I = dicomread(dicominfo(strcat(strcat(rootPath, pathToTestImages), dcmImgFileName)));
        s = regionprops(BW,'centroid');
        centroids = cat(1, s.Centroid);
        mean_centroid = mean(centroids, 1);

        xLeft = mean_centroid(1) - radius;
        yTop = mean_centroid(2) - radius;
        BW = imcrop(BW, [xLeft, yTop, 749, 749]);
        I = imcrop(I, [xLeft, yTop, 749, 749]);
        I = double(I);
        I = I-mean(I(:));
        I = I/std(I(:),0,1);
        maskWriteRoot = 'test_masks_cropped/';
        imgWriteRoot = 'test_mammo_cropped/';
        mkdir(writeDirectory, 'test_masks_cropped');
        mkdir(writeDirectory, 'test_mammo_cropped');
        imwrite(BW, strcat(writeDirectory, strcat(maskWriteRoot, strcat(base_name, '.tif'))));
        imwrite(I, strcat(writeDirectory, strcat(imgWriteRoot, strcat(base_name, '.tif'))));
    end

end