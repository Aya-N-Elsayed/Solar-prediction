clear all;
clc;
addpath(genpath('C:\Users\aeit el rahman\Documents\GitHub\cs519\proj\summer20\ambient_weather\Matlab_grahing'));
filename = 'rf_hyperPWS_7030';
delimiter = ',';
formatSpec = '%s%s%s%s%s%s%s%s%[^\n\r]';
fileID = fopen(filename ,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string',  'ReturnOnError', false); %size 1*9 
fclose(fileID);
%% plotting ytest
time = dataArray{1,1}(3:end,:);
ytest = str2double(dataArray{1,2}(3:end,:));
ts1 = timeseries(ytest,time);
%ts1.Name = filename;
ts1.TimeInfo.Format = 'mm/dd/yyyy HH:MM';       % Set format for display on x-axis.
plot(ts1)
hold on
%% plotting yRf
ytree = str2double(dataArray{1,3}(3:end,:));
ts2 = timeseries(ytree,time);
ts2.Name = filename;
ts2.TimeInfo.Format = 'mm/dd/yyyy HH:MM';       % Set format for display on x-axis.
plot(ts2)
hold on
%% legend
title(filename)
xlabel('time')
ylabel('Solar radiation (W/m^2)')
legend ("Ambient weather","Best model")


