load mristack
BW = mristack < 100;
se = strel('cube',3);
erodedBW = imerode(BW, se);

