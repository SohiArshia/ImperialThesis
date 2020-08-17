

clc
close all
clear

%loading EEG data into struct
for j = 1:13
for i = 1:15
   
filename = sprintf('New _Participant_%d_%d_EEG_both',j,i);
structL(j).Part(i) = load(fullfile('C:\Users\Marjan\Desktop\Project\data\EEGdata',allSID{j},filename));

end
end

%loading Envelope data into struct
for i = 1:15
filenameENV = sprintf('Envelope_both_%d',i);
structL(i).env = load(fullfile('C:\Users\Marjan\Desktop\Project\data\Envelope',filenameENV));
end


%cleaning data so all EEG audiobooks match their respective envelope length 
for j = 1:15
    
    for i = 1:13

    M(i) = size(structL(i).Part(j).EEG,2);

    end
    
    m = min(M);
    
    for i = 1:13
    structL(i).Part(j).EEG = structL(i).Part(j).EEG(:,1:m);
    end
    
    
end

%used for saving cleaned data to be used by Python code, eliminating the
%need to clean in python
% for j = 1:13
%     for i = 1:15
%         EEG = structL(j).Part(i).EEG;
%         
%         savefl = sprintf('Participant_%d_%d',j,i);
%         save(['C:\Users\Marjan\Desktop\Project\data\SubjectLeaveOneOut\' savefl],'-v7','EEG');
%     end
% end 


%%

%looping over all test-train set for particular subject 
tic
%choose subject
for patient = 10
    
    
%choose which audiobook to use as test
for scp = 2

clearvars Env_Im All_Im All_ImT Env_ImT 

%cchoose framesize,channel numbers and overlap/window
Frame_Size = 100;
Num_Channels = 64;
window = 0.25;


%create list to choose training audiobooks
tst = scp;
trn = [] ;
for i = 1:15
    if i ~= tst 
    trn = [trn,i];
    end   
end


%calculate size needed for training set
cont = 1;
for ev = trn
    len = length(structL(ev).env.Envelope);
    tN_Im(cont) = (1 + floor ((len - Frame_Size)/(Frame_Size - round(Frame_Size*window))));
    cont = cont+1;
end
N_Imtot = sum(tN_Im);

Env_Im = zeros(N_Imtot,Frame_Size);
All_Im = zeros(N_Imtot,Num_Channels,Frame_Size);

%calculate size needed for testing set
for ev = scp
%     len = length(structL(ev).env.Envelope)
    TestN_Im = 1 + floor ((len - Frame_Size)/(Frame_Size - round(Frame_Size*window)));
    N_Imtott = sum(TestN_Im);
end

Env_ImT = zeros(N_Imtott,Frame_Size);
All_ImT = zeros(N_Imtott,Num_Channels,Frame_Size);
    





%framing and concatenating for train
z = 0;
%choose subject
for pp = patient
%going through the chosen audibook list made previously 
for j = trn
    Envelope = structL(j).env.Envelope;
    EEG = structL(pp).Part(j).EEG;

    [Row,Column] = size (EEG);
    N_Im = 1 + floor ((length(Envelope) - Frame_Size)/(Frame_Size - round(Frame_Size*window))); %floor((Column)/(Frame_Size));

    for i= 1:N_Im

        z = z + 1;
        Env_Im(z,:) = Envelope(1,(1+(i-1)*(Frame_Size-round(window*Frame_Size))):(i-1)*(Frame_Size-round(window*Frame_Size))+Frame_Size);
        
        All_Im(z,:,:) = EEG(:,(1+(i-1)*(Frame_Size-round(window*Frame_Size))):(i-1)*(Frame_Size-round(window*Frame_Size))+Frame_Size);  

    end  
end
end

%framing and concat for test
z = 0;
for pp = patient

for j = tst
    
   
    Envelope = structL(j).env.Envelope;
    EEG = structL(pp).Part(j).EEG;

    [Row,Column] = size (EEG);
    N_Im = 1 + floor ((length(Envelope) - Frame_Size)/(Frame_Size - round(Frame_Size*window)));

    for i= 1:N_Im

        z = z + 1;
        Env_ImT(z,:) = Envelope(1,(1+(i-1)*(Frame_Size-round(window*Frame_Size))):(i-1)*(Frame_Size-round(window*Frame_Size))+Frame_Size);
        
        All_ImT(z,:,:) = EEG(:,(1+(i-1)*(Frame_Size-round(window*Frame_Size))):(i-1)*(Frame_Size-round(window*Frame_Size))+Frame_Size);  

    end
   
end
end
%name save file
savefl = sprintf('vars_test_%d_subject_%d_25f',scp,patient);
%save
save(['\Desktop\Project\data\SimpleLeaveOneOut\' savefl],'-v7','All_Im','Env_Im','All_ImT','Env_ImT');

end
end
toc
