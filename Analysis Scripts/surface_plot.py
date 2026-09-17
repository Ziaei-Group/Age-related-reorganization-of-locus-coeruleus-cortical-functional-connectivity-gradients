
directory = r''
figure_directory = r''


img_list = [file for file in os.listdir(directory) if file.endswith('combined_G1.nii')]

print(img_list)
name=['']


print(img_list)
print(name)

# get surfaces + sulc maps
surfaces = fetch_fsaverage(density='164k')
lh, rh = surfaces['pial']
sulc_lh, sulc_rh = surfaces['sulc']
gii_sulc_lh = nib.load(str(sulc_lh))
gii_sulc_rh = nib.load(str(sulc_rh))


for filename,tt in zip(img_list,name):
    my_img = os.path.join(directory, filename)
    f=os.path.basename(my_img)
    file_name_without_extension, file_extension = os.path.splitext(f)
    figure_name = figure_directory+'\\'+file_name_without_extension+'.png'
    print(figure_name)

    smooth_anat_img = image.smooth_img(my_img, fwhm=2)
    data_lh, data_rh = mni152_to_fsaverage(smooth_anat_img ,fsavg_density='164k', method='linear')
    p = Plot(lh, rh,brightness=.5)
    p.add_layer({'left': gii_sulc_lh, 'right': gii_sulc_rh}, cmap='binary_r', cbar=False)
    p.add_layer({'left': data_lh, 'right': data_rh},cmap='RdBu_r')

    kws = dict(location='right', draw_border=True, aspect=10, shrink=.3,outer_labels_only=True,
           n_ticks=2,decimals=2, pad=0)
    fig = p.build(cbar_kws=kws)
    fig.axes[0].set_title(tt,fontsize=18)
    fig.show()
    fig.savefig(figure_name,format='png', dpi=300, bbox_inches='tight')