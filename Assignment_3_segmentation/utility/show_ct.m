clear 
close all
clc
D=niftiread('D:\Data\cs-8395-dl\assignment3\Training\img\img0001.nii.gz');% load data
D = squeeze(D);                                % remove singleton dimension
limits = [NaN NaN NaN NaN NaN NaN];
[x, y, z, D] = subvolume(D, limits);           % extract a subset of the volume data

[fo,vo] = isosurface(x,y,z,D);               % isosurface for the outside of the volume
[fe,ve,ce] = isocaps(x,y,z,D);               % isocaps for the end caps of the volume

figure
p1 = patch('Faces', fo, 'Vertices', vo);       % draw the outside of the volume
p1.FaceColor = 'none';
p1.EdgeColor = 'none';

p2 = patch('Faces', fe, 'Vertices', ve, ...    % draw the end caps of the volume
   'FaceVertexCData', ce);
p2.FaceColor = 'interp';
p2.EdgeColor = 'none';

view(-40,24)
daspect([1 1 0.3])                             % set the axes aspect ratio
colormap(gray(100))
box on

camlight(40,40)                                % create two lights 
camlight(-20,-10)
lighting gouraud