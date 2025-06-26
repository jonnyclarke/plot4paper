
"""
Code, inspired by corner.py, to generate pretty corner plots to order
"""
from typing import Union, Tuple
import numpy as np
from scipy.special import erf
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib.ticker import ScalarFormatter

y_formatter = ScalarFormatter(useOffset=False)

_1sigma = erf(1. / np.sqrt(2.))
_2sigma = erf(2. / np.sqrt(2.))
_3sigma = erf(3. / np.sqrt(2.))

_1sigma_2d = 1. - np.exp(-0.5 * (1/1)**2.)
_2sigma_2d = 1. - np.exp(-0.5 * (2/1)**2.)
_3sigma_2d = 1. - np.exp(-0.5 * (3/1)**2.)

beyond_sigma = 1. - _1sigma

_1sigma_low = beyond_sigma/2.
_1sigma_hig = _1sigma_low + _1sigma


def kde_smoothing(x, range_x, nGridPoints):

    """ use KDE smoothing to smear out any infidelities... """
    grid = np.linspace(range_x[0], range_x[1], nGridPoints)

    kde = gaussian_kde(x)

    return grid, kde(grid)


def kde_smoothing_2d(x, y, range_x, range_y, nGridPoints):

    """ use KDE smoothing to smear out any infidelities... """
    gx, gy = np.meshgrid(
        np.linspace(range_x[0], range_x[1], nGridPoints),
        np.linspace(range_y[0], range_y[1], nGridPoints)
    )
    kde = gaussian_kde(np.stack((x, y)))
    gx = np.flipud(gx)
    gy = np.flipud(gy)

    return (
        gx,
        gy,
        kde(np.stack((gx.ravel(), gy.ravel()))).reshape(gx.shape)
    )


class CornerPlot:

    def __init__(
        self,
        grid_shape: Union[int, Tuple[int, int]],
        labelpad=0.4
    ) -> None:
        """
        Brief one-line summary of the function.

        More detailed explanation of the function's behavior (optional).

        Args:
            arg1 (int): Description of arg1.
            arg2 (str): Description of arg2.

        Returns:
            bool: Description of the return value.

        Raises:
            ValueError: Description of when ValueError is raised
            (if applicable).
        """

        if isinstance(grid_shape, int):
            self.__nrows = grid_shape
            self.__ncols = grid_shape

        elif isinstance(grid_shape, tuple):

            if len(grid_shape) != 2:
                raise ValueError("Grid shape tuple must be of length 2")

            for val in grid_shape:
                if not isinstance(val, int):
                    raise ValueError(
                        f"Grid shapes must be of type {type(int)}."
                        f" Input value {val} is of type {type(val)}"
                    )

            (self.__nrows, self.__ncols) = grid_shape

        else:
            raise ValueError(
                "Parameter 'grid_shape' must "
                "be of type 'int' (n --> [n, n]) "
                "or of type 'tuple' ((nrow, ncol) --> [nrow, ncol])"
            )

        self.__lblpad = labelpad

        """
        Initialise the axes
        """
        self.__fig = plt.figure(1)

        self.__gs = gridspec.GridSpec(self.__nrows, self.__ncols)
        self.__gs.update(wspace=0.05, hspace=0.05)
        self.__ax = {}

        self.__LL_CONSTRUCTED = False
        self.__UR_CONSTRUCTED = False

    @property
    def AXES(self):
        return self.__ax

    def __get_nSF(self, val0) -> int:
        val1 = abs(val0)
        if val1 < 10:
            return 1
        else:
            logV = np.log10(val1)
            nSF = np.floor(logV) + 1
            return int(nSF)

    def __default_range(self, x):

        amin = np.amin(x)
        amax = np.amax(x)
        delta = np.ptp(x)

        return [
            amin - 0.1 * delta,
            amax + 0.1 * delta
        ]

    def __format2ndp(self, string, ndp) -> str:

        if ndp == -1:
            s = r"${:+7.1f}$"

        else:
            s = r"${:7."+str(ndp)+"f}$"

        return s.format(string)

    def __histogram(
        self,
        ax,
        x,
        range_x,
        ndp,
        gaussian_prior,
        color,
        nbin,
        quartiles,
        title,
        titleOnBottom=False,
        smooth=False,
        GP_color="k",
        nSmoothingGrid=100
    ) -> None:

        """
        We are constructing the histogram
        """

        if smooth:

            grid, y_n = kde_smoothing(
                x=x,
                range_x=range_x,
                nGridPoints=nSmoothingGrid
            )

            ax.plot(grid, y_n, color=color)

            new_y = np.nanmax(y_n) * 1.1

        else:
            # plot the centroid of the histogram to avoid the blocky edges
            yhist, edges = np.histogram(
                a=x,
                bins=nbin,
                range=range_x,
                density=True
            )
            centroids = (edges[1:] + edges[:-1]) / 2.0
            ax.plot(centroids, yhist, color=color)

            new_y = np.nanmax(yhist) * 1.1

        self.__update_histogram_y_limits(ax, new_y)

        # now we locate the median and the quartiles
        lower, median, upper = np.quantile(
            np.sort(x),
            q=[_1sigma_low, 0.5, _1sigma_hig]
        )

        nsf_val = self.__get_nSF( median )
        nsf_err = np.maximum( self.__get_nSF( upper-median ) , self.__get_nSF( median-lower ) )

        if quartiles[0] : ax.axvline(lower,color=color,ls=":",alpha=0.5)
        if quartiles[1] : ax.axvline(median,color=color,ls=":")
        if quartiles[2] : ax.axvline(upper,color=color,ls=":",alpha=0.5)

        if title and (ndp>=0) :
            vs = "{:"  + str( nsf_val + 1 + ndp ) +"."+str(ndp)+"f}"
            es = "{: >"+ str( nsf_err + 1 + ndp ) +"."+str(ndp)+"f}"
            val = vs.format(median)
            upe = es.format(upper-median)
            lwe = es.format(median-lower)
            upe = upe.rjust(nsf_val + 1 + ndp)
            lwe = lwe.rjust(nsf_val + 1 + ndp)

            if titleOnBottom :
                ax.set_xlabel( r"$ {0}^{{ + {1}  }}_{{ - {2}  }}  $".format(val,upe,lwe) )
            else :
                ax.set_title( r"$ {0}^{{ + {1}  }}_{{ - {2}  }}  $".format(val,upe,lwe),pad=1.5 )

        if gaussian_prior is not None :
            grid = np.linspace( gaussian_prior[0]-5.*gaussian_prior[1], gaussian_prior[0]+5.*gaussian_prior[1], 10000 )
            gaus = np.exp( -0.5 * ( grid - gaussian_prior[0] )**2. / gaussian_prior[1]**2. ) / np.sqrt( 2.*np.pi*gaussian_prior[1]**2. )
            # normalise
            norm = (grid[1] - grid[0])*np.sum(gaus)
            gaus/=norm
            ax.plot(grid,gaus,color=GP_color,ls=":")

    def __2d( self, ax,\
                        x , y,
                        rx, ry,\
                        cmap,\
                        nbin,\
                        yOffset,\
                        color,\
                        use_sigma_regions_2d,\
                        smooth=False,\
                        nSmoothingGrid=100) :

        """
        NB color is only used when one passes 'fadeout'
        to the function and we use it to set alpha to 0
        """

        yoff = y + yOffset

        if not smooth :
            # construct the histogram
            H , xe, ye = np.histogram2d(x,yoff,bins=nbin,range=[ rx , ry ])
            H = np.rot90(H)

        else :

            gx,gy,H = kde_smoothing_2d(x,y,rx,ry,nSmoothingGrid)

            H[ H < np.nanmax(H)/1000. ] = 0.

        if use_sigma_regions_2d :
            """
            we sort the data and plot each pixel a different shade according to the sigma value....
            """

            H_new = np.zeros_like(H).ravel()
            shape = np.shape(H)

            Hflat = H.ravel()
            idx_sort = np.argsort(Hflat)[::-1]

            Hsorted = Hflat[ idx_sort ]
            Hcumsum = np.cumsum(Hsorted)
            Hcumsum /= Hcumsum[-1]

            H_new_sort = np.zeros_like(H_new)

            for lev in [_1sigma_2d, _2sigma_2d] :
                H_new_sort[Hcumsum<=lev] += 1

            baseline = 1
            H_new_sort+=baseline

            # now unsort
            H_new[ idx_sort ] = H_new_sort

            H_new_2d = np.reshape( H_new , shape )
            H_new_2d[H_new_2d==baseline] = np.nan

            if color=="r" :
                cmap = plt.cm.Reds
            elif color=="b" :
                cmap= plt.cm.Blues
            else :
                cmap=plt.cm.Greens

            ax.imshow( H_new_2d, extent = [rx[0],rx[1],ry[0],ry[1]], cmap=cmap,vmin=0,vmax=baseline+3,aspect="auto" )






        elif cmap=="fadeout" :
            # we will use the same color as the plot

            color_map = np.ones(( shape[0],shape[1],4))
            color_map *= color[np.newaxis,np.newaxis,:]

            alphas = H/np.amax(H)
            color_map[...,-1] = alphas

            ax.imshow( color_map, extent = [rx[0],rx[1],ry[0],ry[1]],aspect="auto" )

        else :
            shape = np.shape(H)
            h = H.ravel()
            h[ h==0. ] = np.nan
            H = np.reshape( h , shape )

            ax.imshow( H, extent = [rx[0],rx[1],ry[0],ry[1]], cmap=cmap,aspect="auto" )

    def __update_histogram_y_limits(self,ax,newY) :

        ylims = ax.get_ylim()

        new_low=0
        new_high=np.amax([ ylims[0], newY ])

        ax.set_ylim(new_low,new_high)

    def build_LL_axes( self,\
                        ntheta,\
                        axisRange,\
                        label=None,\
                        ticks=None) :



        if label is None :
            label = [ r"" for i in range(ntheta)]

        if ticks is None :
            ticks = np.array([ None for i in range(ntheta) ])

        assert self.__nrows>=ntheta
        assert self.__ncols>=ntheta
        assert ntheta==len(axisRange)

        self.__LL_ntheta = ntheta

        delta_i = self.__nrows - ntheta

        # define functions to transform from data index to grid index
        def ii(i) :
            return i + delta_i
        def jj(j) :
            return j


        # now we cycle over the samples

        for i in range( ntheta ) :
            for j in range(i+1) :

                """
                i -- cycles over ROWS!!!
                j -- cycles over COLUMNS!!!
                """

                k = "LL"+"_"+str(i)+"_"+str(j)

                # --------------------------------------------------------
                # INITIATE THE SUB-PLOT
                self.__ax[k] = plt.subplot(self.__gs[ii(i),jj(j)])
                # --------------------------------------------------------

                # set tick formatter to avoid irritating plotting defaults
                self.__ax[k].xaxis.set_major_formatter(y_formatter)
                self.__ax[k].yaxis.set_major_formatter(y_formatter)

                # we can set the X-lim straight ahead as ALL subplots respect the same rules
                self.__ax[k].set_xlim(axisRange[j])
                if ticks[j] is not None : self.__ax[k].set_xticks(ticks[j])

                if i==j :
                    """
                        HISTOGRAM PLOTS
                    """
                    self.__ax[k].set_yticks([])
                    self.__ax[k].tick_params(which="major",direction="in",bottom=True,top=True,left=False,right=False)
                else :
                    """
                        NON-HISTOGRAM PLOTS
                    """
                    # we only want the map plots to respect the Y-lims
                    if ticks[i] is not None : self.__ax[k].set_yticks(ticks[i])

                    # set the axis range
                    self.__ax[k].set_ylim(axisRange[i])
                    self.__ax[k].tick_params(which="major",direction="in",bottom=True,left=True,top=True,right=True)

                if (i+1) == ntheta :
                    """
                        Bottom Row
                    """
                    self.__ax[k].xaxis.set_label_coords(x=0.5,y=-self.__lblpad)
                    self.__ax[k].set_xlabel(label[j])
                else :
                    self.__ax[k].set_xticklabels([])



                if (j==0) & (i>0) :
                    """
                        THE n PLOTS W/ Y-AXIS LABELS
                    """
                    # set the y axis label coordinates
                    self.__ax[k].yaxis.set_label_coords(x=-self.__lblpad,y=0.5)
                    # only want Y-labels on the 2D map plots
                    self.__ax[k].set_ylabel(label[i])
                else :
                    # remove all other y-tick labels
                    self.__ax[k].set_yticklabels([])





        self.__LL_CONSTRUCTED=True


    def addData_ll( self,   samples,\
                            ndp=None,\
                            plotrange=None,\
                            gaussian_prior=None,\
                            GP_color="r",\
                            color="blue",\
                            cmap=plt.cm.jet,\
                            nbin=None,\
                            quartiles=[False,True,False],\
                            title=False,\
                            yOffset=0.,\
                            smoothHist=False,\
                            smoothMap =False,\
                            nSmoothingGrid=100,\
                            use_sigma_regions_2d=False ) :

        if not self.__LL_CONSTRUCTED :
            raise Exception("You must first construct the LowerLeft Axes...")

        sample_shape = np.shape(samples)

        nvalue = sample_shape[0]
        ntheta = sample_shape[1]

        assert ntheta==self.__LL_ntheta # ensure the data passed is the correct size for the axes constructed

        if ndp is None :
            ndp = [ 2 for i in range(ntheta) ]

        if plotrange is None :
            plotrange = [ self.__default_range(samples[:,i]) for i in range(ntheta) ]

        if gaussian_prior is None :
            gaussian_prior = np.array([ None for i in range(ntheta) ])


        if nbin is None :
            #> determine the number of bins we should use to average 10 points per bin
            nbin = np.maximum( int( np.sqrt( nvalue / 10 ) ) , 10 )


        # now we cycle over the samples

        for i in range( ntheta ) :
            for j in range(i+1) :
                k = "LL"+"_"+str(i)+"_"+str(j)

                if j==i :
                    # we plot the histogram

                    self.__histogram( self.__ax[k],\
                                        samples[:,i],
                                        plotrange[i],\
                                        ndp[i],\
                                        gaussian_prior[i],\
                                        color=color,\
                                        nbin=nbin,\
                                        quartiles=quartiles,\
                                        title=title,\
                                        smooth=smoothHist,\
                                        titleOnBottom=False,\
                                        GP_color=GP_color,\
                                        nSmoothingGrid=nSmoothingGrid)

                else :
                    # we plot the 2d distribution
                    self.__2d( self.__ax[k],\
                                        samples[:,j], samples[:,i],\
                                        plotrange[j], plotrange[i],\
                                            cmap=cmap,\
                                            nbin=nbin,\
                                            yOffset=yOffset,\
                                            color=color,\
                                            smooth=smoothMap,\
                                            nSmoothingGrid=nSmoothingGrid,\
                                            use_sigma_regions_2d=use_sigma_regions_2d)

                if i == (ntheta-1) :
                    xticks = self.__ax[k].get_xticks()
                    fxticks = [ self.__format2ndp( xtick , ndp[j] ) for xtick in xticks ]
                    self.__ax[k].set_xticklabels( fxticks, rotation = 45)

                if (i>0)&(j==0) :
                    yticks = self.__ax[k].get_yticks()
                    fyticks = [ self.__format2ndp( ytick , ndp[i] ) for ytick in yticks ]
                    self.__ax[k].set_yticklabels( fyticks )







    def build_UR_axes( self,\
                        ntheta,\
                        axisRange,\
                        label=None,\
                        ticks=None) :



        if label is None :
            label = [ r"" for i in range(ntheta)]

        if ticks is None :
            ticks = np.array([ None for i in range(ntheta) ])

        assert self.__nrows>=ntheta
        assert self.__ncols>=ntheta
        assert ntheta==len(label)==len(ticks)==len(axisRange)

        self.__UR_ntheta = ntheta

        delta_i = self.__ncols - ntheta

        # define functions to transform from data index to grid index
        def ii(i) :
            return ntheta - 1 - i
        def jj(j) :
            return self.__ncols - (j + 1)

        # now we cycle over the samples
        for i in range( ntheta ) :
            for j in range(i+1) :

                k = "UR"+"_"+str(i)+"_"+str(j)


                # --------------------------------------------------------
                self.__ax[k] = plt.subplot(self.__gs[ii(i),jj(j)]) # place it where it should be.
                # --------------------------------------------------------

                # set tick formatter to avoid irritating plotting defaults
                self.__ax[k].xaxis.set_major_formatter(y_formatter)
                self.__ax[k].yaxis.set_major_formatter(y_formatter)

                # we can set the X-lim straight ahead as ALL subplots respect the same rules
                self.__ax[k].set_xlim(axisRange[j])
                if ticks[j] is not None : self.__ax[k].set_xticks(ticks[j])

                if i==j :
                    """
                        HISTOGRAM PLOTS
                    """
                    self.__ax[k].set_yticks([])
                    self.__ax[k].tick_params(which="major",direction="in",bottom=True,top=True,left=False,right=False)
                else :
                    """
                        NON-HISTOGRAM plots
                    """
                    # we only want the map plots to respect the Y-lims
                    if ticks[i] is not None : self.__ax[k].set_yticks(ticks[i])
                    # set the axis range
                    self.__ax[k].set_ylim(axisRange[i])
                    self.__ax[k].tick_params(which="major",direction="in",bottom=True,left=True,top=True,right=True)
                #

                if (i+1) == ntheta :
                    """
                        TOP row where we wish to see x ticks and titles
                    """
                    self.__ax[k].xaxis.tick_top()
                    #add the axis label as a title instead
                    # so that the confidence limits can be given as x-label
                    self.__ax[k].set_title(label[j],pad=30)
                else :
                    """
                        OTHER rows
                    """
                    self.__ax[k].set_xticklabels([])

                # define the y labels
                if (j==0)&(i>0) :
                    """
                        THE n PLOTS W/ Y-AXIS LABELS
                    """
                    # set the y axis label coordinates
                    self.__ax[k].yaxis.set_label_coords(x=1.+self.__lblpad*1.6,y=0.5)
                    # add y label to all rows except the first which requires only the first x-label
                    self.__ax[k].set_ylabel(label[i])
                    self.__ax[k].yaxis.tick_right()
                else :
                    # remove all other y-tick labels
                    self.__ax[k].set_yticklabels([])



        self.__UR_CONSTRUCTED=True



    def addData_ur( self,   samples,\
                            ndp=None,\
                            plotrange=None,\
                            gaussian_prior=None,\
                            GP_color="b",\
                            color="red",\
                            cmap=plt.cm.jet_r,\
                            nbin=None,\
                            quartiles=[False,True,False],\
                            title=False,\
                            yOffset=0.,\
                            smoothHist=False,\
                            smoothMap =False,\
                            nSmoothingGrid=100,\
                            use_sigma_regions_2d=False  ) :

        if not self.__UR_CONSTRUCTED :
            raise Exception("You must first construct the UpperRight Axes...")

        sample_shape = np.shape(samples)

        nvalue = sample_shape[0]
        ntheta = sample_shape[1]

        assert ntheta==self.__UR_ntheta # ensure the data passed is the correct size for the axes constructed

        if ndp is None :
            ndp = [ 2 for i in range(ntheta) ]

        if plotrange is None :
            plotrange = [ self.__default_range(samples[:,i]) for i in range(ntheta) ]

        if gaussian_prior is None :
            gaussian_prior = np.array([ None for i in range(ntheta) ])


        if nbin is None :
            #> determine the number of bins we should use to average 10 points per bin
            nbin = np.maximum( int( np.sqrt( nvalue / 10. ) ) , 10 )




        # now we cycle over the samples
        for i in range( ntheta ) :
            for j in range(i+1) :

                k = "UR"+"_"+str(i)+"_"+str(j)

                if j==i :
                    # we plot the histogram

                    self.__histogram(
                        self.__ax[k],
                        samples[:,i],
                        plotrange[i],
                        ndp[i],
                        gaussian_prior[i],
                        color=color,
                        nbin=nbin,
                        quartiles=quartiles,
                        title=title,
                        smooth=smoothHist,
                        titleOnBottom=True,
                        GP_color=GP_color,
                        nSmoothingGrid=nSmoothingGrid
                    )

                else :
                    # we plot the 2d distribution
                    self.__2d(
                        self.__ax[k],\
                        samples[:,j], samples[:,i],\
                        plotrange[j], plotrange[i],\
                        cmap=cmap,\
                        nbin=nbin,\
                        yOffset=yOffset,\
                        color=color,\
                        smooth=smoothMap,\
                        nSmoothingGrid=nSmoothingGrid,
                        use_sigma_regions_2d=use_sigma_regions_2d
                    )
                #
                if i == (ntheta-1) :
                    xticks = self.__ax[k].get_xticks()
                    fxticks = [ self.__format2ndp( xtick , ndp[j] ) for xtick in xticks ]
                    self.__ax[k].set_xticklabels( fxticks, rotation = 45)

                if (i>0)&(j==0) :
                    yticks = self.__ax[k].get_yticks()
                    fyticks = [ self.__format2ndp( ytick , ndp[i] ) for ytick in yticks ]
                    self.__ax[k].set_yticklabels( fyticks )





    @property
    def getFigure(self) :
        return self.__fig, self.__ax








#
#> THE END
#
""" >>> Test script...
fake_data = np.vstack( (np.concatenate((np.random.normal(loc=1.0,scale=0.3,size=500) , np.random.normal(loc=2.,scale=0.1,size=100))),
                        np.concatenate((np.random.normal(loc=1.5,scale=0.5,size=500) , np.random.normal(loc=3.,scale=0.7,size=100))) ))

print(np.shape(fake_data))



useSmoothing=True

qualfig = p4p.qualfig( ncols=2, key="mnras", heightFrac=1.05 )
qualfig.set_label_spaces(side=0.15,top=0.15,bottom=0.15)

plot = hvs.gorgeous_corner(grid_shape=(2,2),labelpad=0.8)


plot.build_LL_axes(ntheta=2,label=label, axisRange=[ (np.amin(x)-0.1*np.ptp(x),np.amax(x)+0.1*np.ptp(x)) for x in fake_data ])#, ticks=ticks)

plot.addData_ll(fake_data.T,quartiles=[True,True,True],\
                smoothHist=True,smoothMap=True,title=True,ndp=[2,2])



fig, ax = plot.getFigure

print("saving....")
#qualfig.save(folder+fname+"__FIT",dpi=1000)

plt.show()

"""
