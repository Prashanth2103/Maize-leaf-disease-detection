
clear all
close all
clc

for ijk = 1:50
    
    I = imread(['Dataset\',num2str(ijk),'.jpg']);
    
    
    % ======= Preprocessing ======= %
    
    % -- Image Resize --
    
    
    IR = imresize(I,[256 256]);
    
    
    % -- Color Transformation -- %
    
    Gray = rgb2gray(IR);
    
    
    % -- Image Enhancement -- %
    
    HIST_EQ = histeq(Gray);
    
    
    % =========== Segmentation ===========
    
    % -- K-means clustering -- %
    
    
    cform = makecform('srgb2lab');
    lab_he = applycform(IR,cform);
    
    ab = double(lab_he(:,:,2:3));
    nrows = size(ab,1);
    ncols = size(ab,2);
    ab = reshape(ab,nrows*ncols,2);
    
    nColors = 2;
    % repeat the clustering 3 times to avoid local minima
    [cluster_idx cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean', ...
        'Replicates',3);
    pixel_labels = reshape(cluster_idx,nrows,ncols);
    
    
    segmented_images = cell(1,3);
    rgb_label = repmat(pixel_labels,[1 1 3]);
    
    for k = 1:nColors
        
        color = IR;
        color(rgb_label ~= k) = 0;
        segmented_images{k} = color;
        
    end
    
    figure,
    subplot(1,3,1)
    imshow(segmented_images{1}),
    title('Cluster 1');
    
    subplot(1,3,2)
    imshow(segmented_images{2}),
    title('Cluster 2');
    
%     subplot(1,3,3)
%     imshow(segmented_images{3}),
%     title('Cluster 3');
    
    INP = input('Enter the cluster number : ')
    
    switch INP
        
        case 1
            SEG = segmented_images{1};
            
        case 2
            SEG = segmented_images{2};
            
%         case 3
%             SEG = segmented_images{3};
            
    end
    
    figure,
    imshow(SEG);
    
    title('Segmented image');
    
    % ========= Feature Extraction ======= %
    
    % -- GLCM - Morphology
    
    glcm = graycomatrix(rgb2gray(SEG),'Offset',[2 0;0 2])
    
    stats = GLCM_Features1(glcm,0);
    
    f1 = stats.autoc;
    f2 = stats.contr;
    
    f3 = stats.corrm;
    f4 = stats.corrp;
    
    f5 = stats.cprom;
    f6 = stats.cshad;
    
    f7 = stats.dissi;
    f8 = stats.energ;
    
    f9 = stats.entro;
    f10 = stats.homom;
    
    f11 = stats.homop;
    f12 = stats.maxpr;
    
    f13 = stats.sosvh;
    f14 = stats.savgh;
    
    f15 = stats.svarh;
    f16 = stats.senth;
    
    f17 = stats.dvarh;
    f18 = stats.denth;
    
    f19 = stats.inf1h;
    f20 = stats.inf2h;
    
    f21 = stats.indnc;
    f22 = stats.idmnc;
    
    GLCM_fea = [f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16...
        f17 f18 f19 f20 f21 f22];
    
    filt = SEG;
    
    if size(filt,3)==3
        im=rgb2gray(filt);
    end
    im=double(filt);
    
    rows=size(im,1);
    cols=size(im,2);
    Ix=im; %Basic Matrix assignment
    Iy=im; %Basic Matrix assignment
    
    % Gradients in X and Y direction. Iy is the gradient in X direction and Iy
    % is the gradient in Y direction
    for i=1:rows-2
        Iy(i,:)=(im(i,:)-im(i+2,:));
    end
    for i=1:cols-2
        Ix(:,i)=(im(:,i)-im(:,i+2));
    end
    
    gauss=fspecial('gaussian',8); %% Initialized a gaussian filter with sigma=0.5 * block width.
    
    angle=atand(Ix./Iy); % Matrix containing the angles of each edge gradient
    angle=imadd(angle,90); %Angles in range (0,180)
    magnitude=sqrt(Ix.^2 + Iy.^2);
    
    % figure,imshow(uint8(angle));
    % figure,imshow(uint8(magnitude));
    
    % Remove redundant pixels in an image.
    angle(isnan(angle))=0;
    magnitude(isnan(magnitude))=0;
    
    feature=[]; %initialized the feature vector
    
    % Iterations for Blocks
    for i = 0: rows/8 - 2
        for j= 0: cols/8 -2
            %disp([i,j])
            
            mag_patch = magnitude(8*i+1 : 8*i+16 , 8*j+1 : 8*j+16);
            %mag_patch = imfilter(mag_patch,gauss);
            ang_patch = angle(8*i+1 : 8*i+16 , 8*j+1 : 8*j+16);
            
            block_feature=[];
            
            %Iterations for cells in a block
            for x= 0:1
                for y= 0:1
                    angleA =ang_patch(8*x+1:8*x+8, 8*y+1:8*y+8);
                    magA   =mag_patch(8*x+1:8*x+8, 8*y+1:8*y+8);
                    histr  =zeros(1,9);
                    
                    %Iterations for pixels in one cell
                    for p=1:8
                        for q=1:8
                            %
                            alpha= angleA(p,q);
                            
                            % Binning Process (Bi-Linear Interpolation)
                            if alpha>10 && alpha<=30
                                histr(1)=histr(1)+ magA(p,q)*(30-alpha)/20;
                                histr(2)=histr(2)+ magA(p,q)*(alpha-10)/20;
                            elseif alpha>30 && alpha<=50
                                histr(2)=histr(2)+ magA(p,q)*(50-alpha)/20;
                                histr(3)=histr(3)+ magA(p,q)*(alpha-30)/20;
                            elseif alpha>50 && alpha<=70
                                histr(3)=histr(3)+ magA(p,q)*(70-alpha)/20;
                                histr(4)=histr(4)+ magA(p,q)*(alpha-50)/20;
                            elseif alpha>70 && alpha<=90
                                histr(4)=histr(4)+ magA(p,q)*(90-alpha)/20;
                                histr(5)=histr(5)+ magA(p,q)*(alpha-70)/20;
                            elseif alpha>90 && alpha<=110
                                histr(5)=histr(5)+ magA(p,q)*(110-alpha)/20;
                                histr(6)=histr(6)+ magA(p,q)*(alpha-90)/20;
                            elseif alpha>110 && alpha<=130
                                histr(6)=histr(6)+ magA(p,q)*(130-alpha)/20;
                                histr(7)=histr(7)+ magA(p,q)*(alpha-110)/20;
                            elseif alpha>130 && alpha<=150
                                histr(7)=histr(7)+ magA(p,q)*(150-alpha)/20;
                                histr(8)=histr(8)+ magA(p,q)*(alpha-130)/20;
                            elseif alpha>150 && alpha<=170
                                histr(8)=histr(8)+ magA(p,q)*(170-alpha)/20;
                                histr(9)=histr(9)+ magA(p,q)*(alpha-150)/20;
                            elseif alpha>=0 && alpha<=10
                                histr(1)=histr(1)+ magA(p,q)*(alpha+10)/20;
                                histr(9)=histr(9)+ magA(p,q)*(10-alpha)/20;
                            elseif alpha>170 && alpha<=180
                                histr(9)=histr(9)+ magA(p,q)*(190-alpha)/20;
                                histr(1)=histr(1)+ magA(p,q)*(alpha-170)/20;
                            end
                            
                            
                        end
                    end
                    block_feature=[block_feature histr]; % Concatenation of Four histograms to form one block feature
                    
                end
            end
            % Normalize the values in the block using L1-Norm
            block_feature=block_feature/sqrt(norm(block_feature)^2+.01);
            
            feature=[feature block_feature]; %Features concatenation
        end
    end
    
    feature(isnan(feature))=0; %Removing Infinitiy values
    
    % Normalization of the feature vector using L2-Norm
    feature=feature/sqrt(norm(feature)^2+.001);
    
    for z=1:length(feature)
        if feature(z)>0.2
            feature(z)=0.2;
            figure(7),subplot(3,4,z),imshow(im{z});axis off;
        end
    end
    
    feature=feature/sqrt(norm(feature)^2+.001);
    
    % toc;
    
    Trainfea(ijk,:) = [GLCM_fea feature];
    
    close all
    clc
    
end

save Trainfea Trainfea