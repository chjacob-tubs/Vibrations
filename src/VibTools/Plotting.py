# -*- coding: utf-8 -*-
#
# This file is part of the
# LocVib 1.3 suite of tools for the analysis for vibrational spectra.
# Copyright (C) 2009-2023 by Christoph R. Jacob and others.
#
#    LocVib is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    LocVib is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with LocVib.  If not, see <http://www.gnu.org/licenses/>.
#
# In scientific publications using the LocVib tools, please cite:
#   Ch. R. Jacob, J. Chem. Phys 130 (2009), 084106.
#
# The most recent version of LocVib is available at
#   http://www.christophjacob.eu/software

import math
import numpy
from matplotlib import pylab as plt
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle
from matplotlib.patches import Arrow
from matplotlib.lines import Line2D


class Plot:

    def __init__(self):
        """
        Plot constructor.
        For more details see the class description/docstring
        """
        self.fig = plt.figure()
        self.ax = None

    def set_figsize(self, figsize):
        """sets figure size."""
        plt.close(self.fig)
        self.fig = plt.figure(figsize=figsize)

    def plot(self):
        """ Creates the figure, but does not draw it """
        pass

    def draw(self):
        """ Creates the figure (for  real). """
        self.ax = self.fig.add_subplot(111)
        self.redraw(self.ax)

    def redraw(self, ax):
        """ Creates the figure inside the given axes. """
        pass

    def save_plot(self, filename, format='pdf'):
        """ Saves the figure to a file. """
        if self.ax is None:
            self.draw()
        self.fig.savefig(filename, format=format, transparent=True)

    def close(self):
        """ Destroys the figure. """
        plt.close(self.fig)


# CombinedPlot has no redraw methods
# -> it cannot be put into another CombinedPlot

class CombinedPlot(Plot):

    def __init__(self):
        """
        CombinedPlot constructor.
        For more details see the class description/docstring
        """
        Plot.__init__(self)

    def plot(self, ncols, nrows, plots):
        plt.close(self.fig)

        w = plots[0].fig.get_figwidth()
        h = plots[0].fig.get_figheight()

        self.fig = plt.figure(figsize=(w*nrows, h*ncols))

        for i, p in enumerate(plots):
            p.close()
            p.redraw(self.fig.add_subplot(ncols, nrows, i + 1))

    def draw(self):
        pass


class HugAnalysisPlot(Plot):

    def __init__(self):
        """
        HugAnalysisPlot constructor.
        For more details see the class description/docstring
        """
        self.fig = plt.figure(figsize=(8.0, 8.0))
        self.ax = None

    def plot(self, matrix, groupnames, maxval=None,
             showsum=True, label_textsize=22):
        self.matrix = matrix
        self.groupnames = groupnames

        if maxval is None:
            self.maxarea = numpy.abs(matrix).max()
        else:
            self.maxarea = maxval

        self.show_sum = showsum

        self.label_textsize = label_textsize

        self.title = None

    def redraw(self, ax):
        ngroups = self.matrix.shape[0]

        ax.set_aspect('equal')

        ax.set_frame_on(False)
        rect = Rectangle((-0.5, -0.5), ngroups+0.5, ngroups+0.5,
                         ec='w', fc='w', fill=True)
        ax.add_patch(rect)

        # make the area of the circles proportional to the matrix elements

        for i in range(ngroups):
            for j in range(i, ngroups):

                area = ((abs(self.matrix[i, j])/self.maxarea)
                        * ((0.4**2)*math.pi))
                rad = math.sqrt(area/math.pi)

                if self.matrix[i, j] < 0.0:
                    fc = 'w'
                else:
                    fc = 'k'

                cir = Circle((j, i), radius=rad, ec='k', fc=fc, lw=2.0)
                ax.add_patch(cir)

        line = Line2D((-0.5, ngroups-0.5), (-0.5, -0.5), color='k')
        ax.add_line(line)
        line = Line2D((ngroups-0.5, ngroups-0.5),
                      (-0.5, ngroups-0.5), color='k')
        ax.add_line(line)

        for i in range(ngroups):
            # horizontal lines
            line = Line2D((i-0.5, ngroups-0.5), (i+0.5, i+0.5), color='k')
            ax.add_line(line)

            # vertical lines
            line = Line2D((i-0.5, i-0.5), (-0.5, i+0.5), color='k')
            ax.add_line(line)

        if self.show_sum:
            ax.text(0.5, ngroups-1.5,
                    'total: %5.2f' % self.matrix.sum(), size=28)

        if self.title:
            ax.text(-1.3, ngroups-0.5, self.title, size=28)

        ax.xaxis.tick_top()
        ax.yaxis.tick_right()

        ax.set_xticks(list(range(ngroups)))
        ax.set_yticks(list(range(ngroups)))

        ax.set_xticklabels(self.groupnames)
        ax.set_yticklabels(self.groupnames)

        for line in ax.xaxis.get_ticklines():
            line.set_markersize(0)
        for line in ax.yaxis.get_ticklines():
            line.set_markersize(0)

        for label in ax.xaxis.get_ticklabels():
            label.set_rotation('vertical')
            label.set_size(self.label_textsize)

        for label in ax.yaxis.get_ticklabels():
            label.set_size(self.label_textsize)

        ax.set_xlim(-0.6, ngroups-0.4)
        ax.set_ylim(ngroups-0.4, -0.6)


class SpectrumPlot(Plot):

    def __init__(self):
        """
        SpectrumPlot constructor.
        For more details see the class description/docstring
        """
        self.fig = plt.figure(figsize=(8.0, 4.0))
        self.ax = None

        self.spectra = []

        self.has_peaklabels = False
        self.peaks = []
        self.has_scalebox = False

        self.center_title = True
        self.title_ypos = 0.95
        self.legend = None

        self.labels = []
        self.boxes = []

    def plot(self, spectra, style=None, lw=None, spectype=None,
             xlims=None, ylims=None, yticks=None):
        self.spectra = spectra

        if style is None:
            self.spectra_style = ['k-'] * len(self.spectra)
        else:
            self.spectra_style = style

        if lw is None:
            self.spectra_lw = [1.0] * len(self.spectra)
        else:
            self.spectra_lw = lw

        if spectype is None:
            self.type = 'General'
        else:
            self.type = spectype

        if self.type == 'General':
            self.title = 'Spectrum'
            self.xlabel = 'wavenumber [cm$^{-1}$]'
            self.ylabel = 'intensity [arb. units]'
        elif self.type == 'IR':
            self.title = 'IR Spectrum'
            self.xlabel = 'wavenumber [cm$^{-1}$]'
            self.ylabel = r'absorption [km/mol]'
        elif self.type == 'Raman':
            self.title = 'Raman Spectrum'
            self.xlabel = 'wavenumber [cm$^{-1}$]'
            self.ylabel = r'scattering factor [${\AA}^4$/a.m.u.]'
        elif self.type == 'ROA':
            self.title = 'ROA Spectrum'
#            self.xlabel = 'wavenumber [cm$^{-1}$]'
            self.xlabel = 'wavenumber / cm$^{-1}$'
#            self.ylabel = r'intensity difference [${\AA}^4$/a.m.u.]'
            yl1 = r'I$_{\rm R}$ $-$ I$_{\rm L}$ /'
            yl2 = r' (${\AA}^4$ a.m.u.$^{-1}$)'
            self.ylabel = yl1 + yl2
        if self.type == 'ROA':
            self.set_figsize((15.0, 7.0))
            self.center_title = False

        if xlims is None:
            self.xlims = (max([sp[0].max() for sp in spectra]),
                          min([sp[0].min() for sp in spectra]))
        else:
            self.xlims = xlims

        if ylims is None:
            self.ylims = (min([sp[1].min() for sp in spectra]),
                          max([sp[1].max() for sp in spectra]))
        else:
            self.ylims = ylims

        self.yticks = yticks

    def scale_range(self, min, max, scale, boxy=None, boxlabel=None):

        for sp in self.spectra:
            indices = numpy.where((sp[0] > min) & (sp[0] < max))
            sp[1][indices] = sp[1][indices] * scale

        if self.has_peaklabels:
            for peak in self.peaks:
                p = peak[0]
                if (p[0] > min) and (p[0] < max):
                    p[1] = p[1] * scale

        if boxy is not None:
            self.has_scalebox = True
            # box: (xmin, xmax, ymin, ymax)
            self.scalebox = (min, max, boxy[0], boxy[1])
            if boxlabel is None:
                self.boxlabel = 'x %f4.1' % scale
            else:
                self.boxlabel = boxlabel

    def add_peaklabels(self, peaks):
        self.has_peaklabels = True
        for p in peaks:
            self.peaks.append([list(p[0]), p[1]])

    def add_box(self, x1, x2, y1, y2, name):
        self.boxes.append(((x1, x2, y1, y2), name))

    def delete_peaklabels(self, peaknums):
        self.peaks = [self.peaks[i] for i in range(len(self.peaks))
                      if i not in peaknums]

    def redraw(self, ax):
        ax.set_autoscale_on(False)
        ax.get_figure().subplots_adjust(hspace=0.3, bottom=0.15, top=0.95)

        self.draw_spectra(ax)
        if self.has_peaklabels:
            self.draw_peaklabels(ax)
        if self.has_scalebox:
            self.draw_scalebox(ax)
        self.draw_boxes(ax)
        self.draw_labels(ax)

        if self.legend:
            ax.legend(self.legend)

    def draw_spectra(self, ax):
        args = []
        for sp, t in zip(self.spectra, self.spectra_style):
            args += [sp[0], sp[1], t]

        lines = ax.plot(*args)

        for line, lw in zip(lines, self.spectra_lw):
            line.set_linewidth(lw)

        ax.set_xlim(self.xlims)
        ax.set_ylim(self.ylims)
        if self.yticks is None:
            yticks = ax.get_yticks()[1:]
            if yticks[-1] > self.ylims[1]:
                yticks = yticks[:-1]
        else:
            yticks = self.yticks

        ax.set_yticks(yticks)

        for label in ax.xaxis.get_ticklabels():
            label.set_size(16)  # 20
        for label in ax.yaxis.get_ticklabels():
            label.set_size(16)

        ytit = self.ylims[0] + self.title_ypos*(self.ylims[1]-self.ylims[0])

        if self.center_title:
            xtit = 0.5*(self.xlims[0]+self.xlims[1])
            ax.text(xtit, ytit, self.title, va='top', ha='center', size=30)
        else:
            xtit = self.xlims[0] + 0.02*(self.xlims[1]-self.xlims[0])
            ax.text(xtit, ytit, self.title, va='top', ha='left', size=30)

        ax.set_xlabel(self.xlabel, size=18)  # 24
        ax.set_ylabel(self.ylabel, size=18)

    def draw_peaklabels(self, ax):
        shift = 0.02 * abs(self.ylims[0]-self.ylims[1])
        for peak in self.peaks:
            p = peak[0]
            if peak[1] > 0:
                ax.text(p[0], min(p[1]+shift, self.ylims[1]),
                        '%4.0f' % p[0], withdash=True,
                        va='center', ha='left', size=16, rotation=70,
                        dashlength=10.0, dashdirection=1)
            else:
                ax.text(p[0], max(p[1]-shift, self.ylims[0]),
                        '%4.0f' % p[0], withdash=True,
                        va='center', ha='right', size=16, rotation=70,
                        dashlength=10.0, dashdirection=0)

    def draw_boxes(self, ax):
        shift = 0.1 * abs(self.ylims[0]-self.ylims[1])
        for b in self.boxes:
            x1, x2, y1, y2 = b[0]
            y1 = y1 - shift
            y2 = y2 + shift
            ax.fill([x2, x2, x1, x1], [y2, y1, y1, y2], fill=False)

    def draw_labels(self, ax):
        shift = 0.02 * abs(self.ylims[0]-self.ylims[1])
        for la in self.labels:
            if la[1] > 0:
                ax.text(la[0], la[1]+shift, la[2],
                        va='bottom', ha='left', size=14)
            else:
                ax.text(la[0], la[1]-shift, la[2],
                        va='top', ha='left', size=14)

    def draw_scalebox(self, ax):
        x1, x2, y1, y2 = self.scalebox
        ax.fill([x2, x2, x1, x1], [y2, y1, y1, y2], fill=False)

        if (self.xlims[0] < self.xlims[1]):
            xlab = min(x1, x2)
        else:
            xlab = max(x1, x2)
        ylab = max(y1, y2)

        ax.text(xlab, ylab*1.05, self.boxlabel,
                va='bottom', ha='left', size=18)


class ModesPlot(Plot):

    def __init__(self):
        """
        ModesPlot constructor.
        For more details see the class description/docstring
        """
        self.fig = plt.figure(figsize=(7.0, 10.0))
        self.ax = None

    def plot(self, modes, freqs, ints, locfreqs, modelabels=None, range=None):
        self.modes = modes
        self.freqs = freqs
        self.ints = ints
        self.locfreqs = locfreqs

        if range is None:
            self.range = (freqs.max(), freqs.min())
        else:
            self.range = range

        if modelabels is None:
            self.modelabels = list(range(len(freqs)))
        else:
            self.modelabels = modelabels

        self.title = ''
        self.xlabel = ''
        self.ylabel = 'wavenumber [cm$^{-1}$]'

        self.yticks = None

    def redraw(self, ax):
        ax.set_frame_on(False)
        ax.set_autoscale_on(False)

        numfreqs = len(self.freqs)

        xmin = -numfreqs
        xmax = +numfreqs

        asize = abs(self.range[1]-self.range[0]) / 20.0

        rect = Rectangle((xmin-1.0, self.range[1]), 2*xmax+6.0,
                         abs(self.range[1]-self.range[0]),
                         ec='w', fc='w', fill=True)
        ax.add_patch(rect)

        line = Line2D((xmin-1.0, xmax+5.0),
                      (self.range[0], self.range[0]), color='k')
        ax.add_line(line)
        line = Line2D((xmin-1.0, xmin-1.0),
                      (self.range[0], self.range[1]), color='k')
        ax.add_line(line)

        for i in range(len(self.locfreqs)):
            line = Line2D((xmin-0.5, -0.5),
                          (self.locfreqs[i], self.locfreqs[i]), color='k')
            ax.add_line(line)

            arrow = Arrow(xmin+i, self.locfreqs[i],
                          0, -asize, width=0.5, fc='k', ec='k')
            ax.add_patch(arrow)

        for i in range(len(self.freqs)):
            line = Line2D((0.5, xmax+0.5),
                          (self.freqs[i], self.freqs[i]), color='k')
            ax.add_line(line)

            ax.text(xmax+1.0, self.freqs[i], 'int:', va='bottom', ha='left')
            ax.text(xmax+4.1, self.freqs[i], '%7.1f' % self.ints[i],
                    va='bottom', ha='right')

            for j in range(len(self.freqs)):
                arrow = Arrow(j+1, self.freqs[i], 0, self.modes[i, j]*asize,
                              width=0.5, fc='k', ec='k')
                ax.add_patch(arrow)

        ax.set_xlim((xmin-1.0, xmax+5.1))
        ax.set_ylim((self.range[0]+0.1, self.range[1]-0.1))

        ax.xaxis.tick_bottom()
        ax.yaxis.tick_left()

        ax.set_xticks(list(range(xmin, 0))+list(range(1, xmax+1)))
        ax.set_xticklabels(self.modelabels + self.modelabels)

        if self.yticks is not None:
            ax.set_yticks(self.yticks)

        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)

        ax.text(0.5*(xmin-1), self.range[0]-0.5*asize, 'localized modes',
                va='center', ha='center')

        ax.text(0.5*(xmax+1), self.range[0]-0.5*asize, 'coupled modes',
                va='center', ha='center')

        ax.text(0.0, self.range[1]+0.5*asize, self.title,
                va='center', ha='center', size=24)
