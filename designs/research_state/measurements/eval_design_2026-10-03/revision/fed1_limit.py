"""F-ED-1's detection limit: the excess (non-BT) per-edge SD a single legacy ladder could detect at 80 % power
with the Pearson chi-square at alpha = 0.05 (chi-square approximation): K = 20 nodes, 141 edges, 100 unmirrored games per
edge (N0's shape), per-edge binomial SE at 1/2 = 5 pp. Under a random-effects alternative T ~ (1 + tau^2/SE^2) chi2_df."""
from scipy.stats import chi2
from scipy.optimize import brentq
df=141-19; se=5.0
for pw in (0.5,0.8):
    r=brentq(lambda r: chi2.sf(chi2.ppf(.95,df)/(1+r),df)-pw,1e-6,10)
    print(f"power {pw}: tau = {se*r**.5:.2f} pp (phi = {1+r:.3f})")
