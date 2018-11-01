% preprocess training data
training_file_path = '/Final_Training/Images/';

imadjust_file_path = '/Final_Training/imadjust/';
histeq_file_path = '/Final_Training/histeq/';
adapthisteq_file_path = '/Final_Training/adapthisteq/';
conorm_file_path = 'Final_Training/conorm_file_path/';

kernel_size = [5, 5];
sigma1 = 0.5;
sigma2 = 5;
kernel1 = fspecial('Gaussian', kernel_size, sigma1);
kernel2 = fspecial('Gaussian', kernel_size, sigma2);

classes = dir(training_file_path + '00*');
for class = classes'
    mkdir(imadjust_file_path + class.name);
    mkdir(histeq_file_path + class.name);
    mkdir(adapthisteq_file_path + class.name);
    mkdir(conorm_file_path + class.name);
    
    images = dir(training_file_path + class.name + '/*.ppm');
    for image = images'
        img = imread(training_file_path + class.name + '/' + image);
        img_lab = rgb2lab(img);
        
        max_luminosity = 100;
        
        img_imadjust = img_lab;
        img_imadjust(:, :, 1) = imadjust(img_lab(:, :, 1)/max_luminosity) * max_luminosity;
        img_imadjust = lab2rgb(img_imadjust);
        imwrite(img_imadjust, imadjust_file_path + class.name + '/' + image);
        
        img_histeq = img_lab;
        img_histeq(:, :, 1) = histeq(img_lab(:, :, 1)/max_luminosity) * max_luminosity;
        img_histeq = lab2rgb(img_histeq);
        imwrite(img_histeq, histeq_file_path + class.name + '/' + image);
        
        img_adapthisteq = img_lab;
        img_adapthisteq(:, :, 1) = adapthisteq(img_lab(:, :, 1)/max_luminosity) + max_luminosity;
        img_adapthisteq = lab2rgb(img_adapthisteq);
        imwrite(img_adapthisteq, adapthisteq_file_path + class.name + '/' + image);
        
        gauss1 = imfilter(img, kernel1, 'replicate');
        gauss2 = imfilter(img, kernel2, 'replicate');
        img_conorm = gauss2 - gauss1;
        imwrite(img_conorm, conorm_file_path + class.name + '/' + image);
    end
end