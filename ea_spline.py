import os, sys, argparse
import pandas as pd
import numpy as np
 
def ea_spline(data:pd.DataFrame, var:str, 
    depths:list=[0,5,15,30,60,100,200], lam:float=0.1,
    vlow:int=0, vhigh:int=1000, showProgress:bool=True
):
    if isinstance(data, pd.DataFrame):
        idxs = [0,1,2]
        if isinstance(var, int):
            idxs.append(var)
        else:
            vidx = data.columns.get_loc(var)
            idxs.append(vidx)
        data = data.iloc[:,idxs]
    # convert to column vector
    depth_arr = np.array(depths)[np.newaxis]
    max_dep = np.max(depth_arr)

    # overall variance of the variable
    variance = np.power(0.05 * np.mean(data.iloc[:,3]), 2)
    
    # group data
    gp_data = data.groupby(data.iloc[:,0])
    # length of lambda array
    nlam = 1

    # array of the continous splines for each data point
    mat_yfit = np.empty( (len(gp_data), len(np.arange(0,max_dep))) )


    # array of the averaged values where splines are fitted.
    yavg = np.full( (len(gp_data), depth_arr.shape[1]), np.nan)

    # array of the sum of square errors for each lamda iteration where working profile is stored
    sse_mat = np.empty( (1, nlam) )

    ## array of the sum of square errors for each lambda iteration where each profile is stored
    sset_mat = np.empty( (len(gp_data), nlam) )

    # profile ids
    df_id = pd.DataFrame(columns=pd.Series(data.columns))
    
    # data frame for observations and spline predictions
    davg = pd.DataFrame(index=np.arange(data.shape[0]), columns=list(df_id) + ['predicted', 'FID'])

    # fit splines by profile
    old_prog = 1
    counter = 0
    fid = 1
    for profile_label in gp_data.groups.keys():
        profile = gp_data.get_group(profile_label)
        df_id.loc[fid-1] = pd.Series([profile_label, 0, 0, 0])
        df_id.iloc[fid-1,0] = profile_label
        
        # manipulate the profile data to the required form
        ir = np.array(np.arange(profile.shape[0]))[np.newaxis]
        # upper
        u = np.array(profile.iloc[:, 1])[np.newaxis]
        # lower
        v = np.array(profile.iloc[:, 2])[np.newaxis]
        # variable
        y = np.array(profile.iloc[:, 3])[np.newaxis]
        n = y.shape[1]

        # handle single observation profiles
        if n == 1:
            davg.iloc[counter:(counter + profile.shape[0]), 0:4] = np.array(profile)
            davg.iloc[counter:(counter + profile.shape[0]), 4] = np.array(y)
            davg.iloc[counter:(counter + profile.shape[0]), 5] = np.full((profile.shape[0]), fid)

            # extrapolate values onto yfit
            xfit = np.arange(max_dep)[np.newaxis].T
            nj = np.max(v)
            nj = (nj * int(nj <= max_dep)) + (max_dep * int(nj > max_dep))
            yfit = np.full((max_dep, 1), np.nan)
            yfit[0:nj+1,0] = y[0,0]
            if nj < max_dep:
                yfit[(nj+1):max_dep,:] = -9999
            mat_yfit[fid-1,:] = yfit[:,0]

            # averages of the spline at specified depths
            ndi = depth_arr.shape[1] - 1
            di = depth_arr
            for cj in range(ndi):
                xd1 = di[0,cj]
                xd2 = di[0,cj+1]
                if nj>=xd1 and nj<=xd2:
                    xd2 = nj - 1
                ydat = yfit[xd1:xd2,0]
                if np.isnan(ydat).all():
                    yavg[fid-1,cj] = np.nan
                else:
                    yavg[fid-1,cj] = np.nanmean(ydat)
                yavg[fid-1,cj+1] = np.max(v)
            counter += profile.shape[0]
        else:
            davg.iloc[counter:(counter + profile.shape[0]), 0:4] = np.array(profile)
            davg.iloc[counter:(counter + profile.shape[0]), 5] = np.full((profile.shape[0]), fid)
            
            # estimate spline parameters
            nplus1 = n+1
            nless1 = n-1
            delta = v-u  # depths of each layer
            dif = np.column_stack([u[:,1:], u[:,-1]]) - v

            # create a diagonal matrix (n-1,n-1) of 1's with zeros elsewhere
            r = np.zeros((nless1, nless1))
            np.fill_diagonal(r, 1)
            # correct upper diagonal
            for upper_diag in range(nless1-1):
                r[upper_diag, upper_diag+1] = 1
            # create a diagonal matrix of layer depths to premultiply the current r
            d2 = np.zeros((nless1, nless1))
            np.fill_diagonal(d2, delta[:,1:])
            # then premultiply and add the transpose; this gives half of r
            r = np.matmul(d2, r)
            r = r + r.T
            # then create a new diagonal matrix for differences to add to the diagonal
            d1 = np.zeros((nless1, nless1))
            np.fill_diagonal(d1, delta[:,:-1])
            
            d3 = np.zeros((nless1, nless1))
            np.fill_diagonal(d3, dif[:,:-1])

            r = r + (2*d1) + (6*d3)

            # create the (n-1)xn matrix q
            q = np.zeros((n, n))
            np.fill_diagonal(q, -1)
            for upper_diag in range(nless1):
                q[upper_diag, upper_diag+1] = 1
            q = q[:-1, :]
            dim_mat = q

            try:
                rinv = np.linalg.inv(r)
            except np.linalg.LinAlgError:
                continue
            
            # identity matrix i
            imat = np.zeros((n, n))
            np.fill_diagonal(imat, 1)

            # create the matrix coefficent z
            pr_mat = np.full((n, nless1), 6*n*lam) #np.zeros((n, nless1))
            fdub = np.matmul(pr_mat * dim_mat.T, rinv)
            z = np.matmul(fdub, dim_mat)+imat
            
            # solve for the fitted layer means
            sbar = np.linalg.solve(z, y.T)
            davg.iloc[counter:(counter + profile.shape[0]), 4] = sbar.flatten()
            counter += profile.shape[0]
            
            # calculate the fitted value at the knots
            b = 6 * np.matmul(np.matmul(rinv, dim_mat), sbar)
            b0 = np.insert(b, 0, 0).reshape((profile.shape[0], 1))
            b1 = np.append(b, 0).reshape((profile.shape[0], 1))
            tdeltaT = 2*delta.T
            gamma = np.divide((b1-b0), tdeltaT, out=np.full(tdeltaT.shape,np.nan), where=tdeltaT != 0)
            alpha = sbar - (b0 * delta.T / 2) - (gamma * (delta.T ** 2/3))

            # fit the spline
            xfit = np.arange(max_dep)[np.newaxis].T
            nj = np.max(v)
            nj = (nj * int(nj <= max_dep)) + (max_dep * int(nj > max_dep))
            yfit = np.array(np.arange(max_dep), dtype=np.float64)[np.newaxis].T
            for k in range(nj):
                xd = xfit[k]
                if xd < u[0,0]:
                    p = alpha[0,0]
                else:
                    for i in range(n):
                        if i < nless1:
                            tf2 = int(xd>v[0,i] and xd<u[0,i+1])
                        else:
                            tf2 = 0
                        if xd>=u[0,i] and xd<=v[0,i]:
                            xde = float((xd+1) - u[0,i])
                            p = alpha[i,0] + b0[i,0] * xde + gamma[i,0] * xde**2
                        elif tf2:
                            phi = alpha[i+1,0] - b1[i,0] * (u[0,i+1]-v[0,i])
                            p = phi + b1[i,0] * ((xd+1)-v[0,i])
                    
                yfit[k,0] = p
            if nj < max_dep:
                yfit[(nj):max_dep,:] = np.nan
            
            yfit[np.where(yfit > vhigh)] = vhigh
            yfit[np.where(yfit < vlow)] = vlow
            mat_yfit[fid-1, :] = yfit[:,0]

            # averages of the spline at specified depths
            ndi = depth_arr.shape[1] - 1
            di = depth_arr
            for cj in range(ndi):
                xd1 = di[0,cj]
                xd2 = di[0,cj+1]
                if nj>=xd1 and nj<=xd2:
                    xd2 = nj - 1
                ydat = yfit[xd1:xd2,0]
                if np.isnan(ydat).all():
                    yavg[fid-1,cj] = np.nan
                else:
                    yavg[fid-1,cj] = np.nanmean(ydat)
                yavg[fid-1,cj+1] = np.max(v)
            
            ## calcualte the error between the observed and fitted values
            ## Wahba's estimate of the residual variance sigma^2
            ssq = np.sum((y.T - sbar)**2)
            ident = np.zeros(z.shape)
            np.fill_diagonal(ident, 1)
            g = np.linalg.solve(z, ident)
            ei = np.linalg.eigvals(g)
            degf = n - sum(ei)
            #sig2w = ssq / degf
            # calculate the Carter and Eagleson estimate of residual variance
            #degfc = n - 2*np.sum(ei) + np.sum(ei**2)
            #sig2c = ssq / degfc
            # calculate the estimate of the true mean squared error
            tmse = ssq / n - 2 * variance * degf / n + variance
            sset_mat[fid-1] = tmse

        if showProgress:
            prog = int(100 * fid / len(gp_data))
            if prog != old_prog:
                print('Progress: {}%.'.format(prog))
                old_prog = prog
        fid += 1

    # prepare output
    yavgdf = pd.DataFrame(yavg)
    yavgdf.insert(0, column='id', value=list(gp_data.groups.keys()))
    jmat = ['id'] + ["{}-{} cm".format(depths[i], depths[i+1]) for i in range(len(depths)-1)] + ['soil depth']
    yavgdf.columns = jmat

    return (yavgdf, davg, mat_yfit.T, sset_mat)

def main(argv):
    parser = argparse.ArgumentParser(usage=__doc__, add_help=False)
    parser.conflict_handler="resolve"
    
    parser.add_argument("--data", dest="INPUT_PATH", required=True, type=str,
                        help="Path to the input data. CSV format with the first 3 columns being the SiteID, followed by the upper and lower horizon depths (cm).")

    parser.add_argument("--var", dest="VARIABLE", required=True, type=str,
                        help="Column heading of the variable to harmonize.")

    parser.add_argument("--output", dest="OUTPUT_PATH", required=True, type=str,
                        help="Path to write an output file.")

    parser.add_argument("--depths", dest="DEPTH_STR", required=False, type=str,
                        help="Comma separated depth interval integers in cm. Defaults to GlobalSoilMap depths.")
    
    parser.add_argument("--lam", dest="LAMBDA", required=False, type=float, default=0.1,
                        help="Equal area spline lambda parameter.")
    
    parser.add_argument("--vhigh", dest="VHIGH", required=False, type=int, default=1000,
                        help="Sets the maximum of the fitted variable range.")
    
    parser.add_argument("--vlow", dest="VLOW", required=False, type=int, default=0,
                        help="Sets the minimum of the fitted variable range.")
    
    parser.add_argument("--show_prog", dest="PROG", required=False, type=bool, default=False,
                        help="Set to True to print progress updates to the standard out.")

    parser.add_argument("-?", "-h", "--help", action="help", 
                        help="Display the program help and ends the script")									

    #Stocking the parameters into arguments
    args, _ = parser.parse_known_args(argv)
    
    # create variables from arguments
    DATA_PATH = args.INPUT_PATH
    if not (os.path.isfile(DATA_PATH) and os.path.splitext(DATA_PATH)[1].lower() == '.csv'):
        raise Exception('Please enter the full path to the input .csv data.')
    VAR = args.VARIABLE
    if 'DEPTH_STR' in args and args.DEPTH_STR != None:
        DEPTHS = [int(d) for d in args.DEPTH_STR.split(',') if d.isnumeric()]
    else:
        DEPTHS = [0,5,15,30,60,100,200]
    LAMBDA = args.LAMBDA
    VLOW = args.VLOW
    VHIGH = args.VHIGH
    PROGRESS = args.PROG
    OUTPUT = os.path.splitext(args.OUTPUT_PATH)[0] + '.csv'
    if not os.path.isdir(os.path.split(OUTPUT)[0]):
        raise Exception('Please enter a valid directory to write the output data.')

    DATA = pd.read_csv(DATA_PATH)
    if not VAR in  DATA.columns:
        raise Exception('Please enter the variable as it appears in the input .csv header.')
    
    ea_spline(DATA, VAR, DEPTHS, LAMBDA, VLOW, VHIGH, PROGRESS)[0].to_csv(OUTPUT)

if __name__ == "__main__":
    main(sys.argv[1:])
