% preprocess training data
training_file_path = '~/Projects/ciresan-meier-masci-schmidhuber-2012/training_set/original/';

imadjust_file_path = '~/Projects/ciresan-meier-masci-schmidhuber-2012/training_set/imadjust/';
histeq_file_path = '~/Projects/ciresan-meier-masci-schmidhuber-2012/training_set/histeq/';
adapthisteq_file_path = '~/Projects/ciresan-meier-masci-schmidhuber-2012/training_set/adapthisteq/';

classes = dir(strcat(training_file_path, '00*'));
for class = classes'
    class_name = class.name;%string(class);
    
    feedback = ['processing class: ', class_name];
    disp(feedback);
    
    mkdir(strcat(imadjust_file_path, class_name));
    mkdir(strcat(histeq_file_path, class_name));
    mkdir(strcat(adapthisteq_file_path, class_name));
    
    images = dir(strcat(training_file_path, class_name, '/*.ppm'));
    for image = images'
        image_name = image.name;
        
        img = imread(strcat(training_file_path, class_name, '/', image_name));
        
        img = img/255;
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
    end
end