function process_list_beam(inList)
% Function to process list of raw files and make the corresponding feature
% files
% inList has 3 arguments

maxNumCompThreads(3);

[xaa,xab,xac] = textread (inList,'%s %s %s');

for I = 1 : length(xaa)
    disp (['Processing ' num2str(I) ' of ' num2str(length(xaa)) ' files...']);
    beamformit(xaa{I},xab{I},xac{I});
end


