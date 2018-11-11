function [] = normalize(path)
% normalize training images
training_file_path = fullfile(path, 'original');

imadjust_file_path = fullfile(path, 'imadjust');
histeq_file_path = fullfile(path, 'histeq');
adapthisteq_file_path = fullfile(path, 'adapthisteq');

classes = dir(fullfile(training_file_path, '00*'));
for class = classes'
    class_name = class.name;
    
    feedback = ['normalizing class ', class_name];
    disp(feedback);

    mkdir(fullfile(imadjust_file_path, class_name));
    mkdir(fullfile(histeq_file_path, class_name));
    mkdir(fullfile(adapthisteq_file_path, class_name));

    images = dir(fullfile(training_file_path, class_name, '*.ppm'));
    for image = images'
        image_name = image.name;

        img = imread(fullfile(training_file_path, class_name, image_name));

        img_lab = rgb2lab(img);

        max_luminosity = 100;

        img_imadjust = img_lab;
        img_imadjust(:, :, 1) = imadjust(img_lab(:, :, 1)/max_luminosity) * max_luminosity;
        img_imadjust = lab2rgb(img_imadjust);
        imwrite(img_imadjust, fullfile(imadjust_file_path, class_name, image_name));

        img_histeq = img_lab;
        img_histeq(:, :, 1) = histeq(img_lab(:, :, 1)/max_luminosity) * max_luminosity;
        img_histeq = lab2rgb(img_histeq);
        imwrite(img_histeq, fullfile(histeq_file_path, class_name, image_name));

        img_adapthisteq = img_lab;
        img_adapthisteq(:, :, 1) = adapthisteq(img_lab(:, :, 1)/max_luminosity) + max_luminosity;
        img_adapthisteq = lab2rgb(img_adapthisteq);
        imwrite(img_adapthisteq, fullfile(adapthisteq_file_path, class_name, image_name));
    end
end
end