clear all;
clc;
addpath(genpath('C:\Users\aeit el rahman\Documents\GitHub\cs519\proj\summer20\ambient_weather\Matlab_grahing'));
filename = 'TRPWSTS0';
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
%% plotting ytree
ytree = str2double(dataArray{1,3}(3:end,:));
ts2 = timeseries(ytree,time);
ts2.Name = filename;
ts2.TimeInfo.Format = 'mm/dd/yyyy HH:MM';       % Set format for display on x-axis.
plot(ts2)
hold on
%% plotting yRf
yRf = str2double(dataArray{1,4}(3:end,:));
ts3 = timeseries(yRf,time);
ts3.Name = filename;
ts3.TimeInfo.Format = 'mm/dd/yyyy HH:MM';       % Set format for display on x-axis.
plot(ts3)
hold on
%% plotting ysvm
ysvm = str2double(dataArray{1,5}(3:end,:));
ts4 = timeseries(ysvm,time);
ts4.Name = filename;
ts4.TimeInfo.Format = 'mm/dd/yyyy HH:MM';       % Set format for display on x-axis.
plot(ts4)
hold on
%% plotting ybr
ybr = str2double(dataArray{1,6}(3:end,:));
ts5 = timeseries(ybr,time);
ts5.Name = filename;
ts5.TimeInfo.Format = 'mm/dd/yyyy HH:MM';       % Set format for display on x-axis.
plot(ts5)
hold on
%% plotting ylr
ylr = str2double(dataArray{1,7}(3:end,:));
ts6 = timeseries(ylr,time);
ts6.Name = filename;
ts6.TimeInfo.Format = 'mm/dd/yyyy HH:MM';       % Set format for display on x-axis.
plot(ts6)
hold on
%% plotting ympl
ympl = str2double(dataArray{1,8}(3:end,:));
ts7 = timeseries(ympl,time);
ts7.Name = filename;
ts7.TimeInfo.Format = 'mm/dd/yyyy HH:MM';       % Set format for display on x-axis.
plot(ts7)
hold on
%% legend
title('PWS (67865, 15) testing on PWS20200921 (289, 15)')
xlabel('time')
ylabel('Solar radiation (W/m^2)')
legend ("Ambient weather","Decision tree","Random forest","SVR linear","Bayesian Ridge","Linear regression","MPL NN")


