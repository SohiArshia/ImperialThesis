function [data,iB] = LoadEEG(c)
iB = 1;
C = c{1};
f = fullfile(C{1},C{2});
d = load(f);
data = transpose(d.EEG);
end