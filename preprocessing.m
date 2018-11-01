% preprocess training data
training_file_path = '~/Projects/ciresan-meier-masci-schmidhuber-2012/Final_Training/Images/';

imadjust_file_path = '~/Projects/ciresan-meier-masci-schmidhuber-2012/Final_Training/imadjust/';
histeq_file_path = '~/Projects/ciresan-meier-masci-schmidhuber-2012/Final_Training/histeq/';
adapthisteq_file_path = '~/Projects/ciresan-meier-masci-schmidhuber-2012/Final_Training/adapthisteq/';
conorm_file_path = '~/Projects/ciresan-meier-masci-schmidhuber-2012/Final_Training/conorm/';

kernel_size = [5, 5];
sigma1 = 0.5;
sigma2 = 5;
kernel1 = fspecial('Gaussian', kernel_size, sigma1);
kernel2 = fspecial('Gaussian', kernel_size, sigma2);

classes = dir(strcat(training_file_path, '00*'));
for class = classes'
    class_name = class.name;%string(class);
    
    feedback = ['processing class: ', class_name];
    disp(feedback);
    
    mkdir(strcat(imadjust_file_path, class_name));
    mkdir(strcat(histeq_file_path, class_name));
    mkdir(strcat(adapthisteq_file_path, class_name));
    mkdir(strcat(conorm_file_path, class_name));
    
    images = dir(strcat(training_file_path, class_name, '/*.ppm'));
    for image = images'
        image_name = image.name;
        
        img = imread(strcat(training_file_path, class_name, '/', image_name));
        img_lab = rgb2lab(img);
        
        max_luminosity = 100;
        
        img_imadjust = img_lab;
        img_imadjust(:, :, 1) = imadjust(img_lab(:, :, 1)/max_luminosity) * max_luminosity;
        img_imadjust = lab2rgb(img_imadjust);
        imwrite(img_imadjust, strcat(imadjust_file_path, class_name, '/', image_name));
        
        img_histeq = img_lab;
        img_histeq(:, :, 1) = histeq(img_lab(:, :, 1)/max_luminosity) * max_luminosity;
        img_histeq = lab2rgb(img_histeq);
        imwrite(img_histeq, strcat(histeq_file_path, class_name, '/', image_name));
        
        img_adapthisteq = img_lab;
        img_adapthisteq(:, :, 1) = adapthisteq(img_lab(:, :, 1)/max_luminosity) + max_luminosity;
        img_adapthisteq = lab2rgb(img_adapthisteq);
        imwrite(img_adapthisteq, strcat(adapthisteq_file_path, class_name, '/', image_name));
        
        gauss1 = imfilter(img, kernel1, 'replicate');
        gauss2 = imfilter(img, kernel2, 'replicate');
        img_conorm = gauss2 - gauss1;
        imwrite(img_conorm, strcat(conorm_file_path, class_name, '/', image_name));
    end
end