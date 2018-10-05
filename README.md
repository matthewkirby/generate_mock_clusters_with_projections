# generate_mock_clusters_with_projections
A script to quickly draw observed redmapper richness given a true redmapper richness.

# To use the code

1. Initialize an instance of the Converter class `conv = Converter()`
2. Call the `draw_from_cdf` method with a list of true richnesses and true redshifts `lamobs_list = conv.draw_from_cdf(lamtrue_list, ztrue_list)`
3. The resulting array is a list of observed richnesses!

# Warnings and caveats

- This code is only valid for clusters with z in [0.1, 0.3) and true richness in [1, 300). Attempting to use it outside of these bounds will cause an error.
- This code gets its speed by pre-computing CDFs on a 2d grid of z and lamtrue. It takes several hours to compute these. I have included a .pkl file that contains the CDF grid. Please keep in mind that a pickle file may contain anything and only open pickle files supplied to you by someone that you trust!
- If you wish to compute your own grid, you will only need to do it once. The code will then save the grid as a pickle file that it will then access at future run times.

The code is supplied **as is** and all use is at **your own risk**.
