
define(function(require) {

    d3 = require('/nbextensions/phy/static/d3.js');

    function D3ClusterWidget(clusterView) {
        this.view = clusterView;
        this.width = 95;
        this.height = 95;
        this.onSelected = function(selection) {};

        this.selected = function() {
            var res = [];

            d3.select(this.view).selectAll('.cluster').each(function(dl, i) {
                if (dl.selected) {
                    res.push(dl.id)
                }
            });
            return res;
        }

        this.redraw = function(clusters, displayAttr)
        {
            var barWidth = 1;

            var cluster = d3.select(this.view).selectAll(".cluster").data(clusters, function(d) { return d.id; });
            var that = this;
            var clusterEnter = cluster.enter()
                                .insert("div")
                                .attr("class", function (d) {return (d.selected) ? "cluster sel" : "cluster";})
                                .style("fill-opacity", 0)
                                .on('mousedown', function(d) {
                                    that.clusterClicked(d, this);
                                    that.redraw(clusters, displayAttr);
                                    that.onSelected(that.selected());
                                });

            var lineFunction = d3.svg.line()
                                 .x(function(d, i) { return that.width-(i*barWidth); })
                                 .y(function(d, i) { return that.height-d; })
                                 .interpolate("step-after");

            var svgContainer = clusterEnter.append("svg")
                                .attr("width", that.width)
                                .attr("height", that.height);


            svgContainer.append("path")
                        .datum(function(d) { barWidth = that.width / d.ccg.length; return d.ccg;})
                        .attr("d", function(d) {return lineFunction(d);})
                        .attr("stroke", "black")
                        .attr("stroke-width", 0)
                        // .attr("fill", "black")

            for (var i = 0; i < displayAttr.length; i++) {
                clusterEnter.append("div")
                    .attr("class", "clusterinfo " + displayAttr[i])
                    .text(function(d) {return d[displayAttr[i]];});
            }

            //
            // Clusters to update
            //
            var clusterUpdate = d3.transition(cluster)
                .style("fill-opacity", 1)
                .attr("class", function (d) {return (d.selected) ? "cluster sel" : "cluster";});

            //
            // Clusters to remove
            //
            var clusterExit = d3.transition(cluster.exit())
                .style("fill-opacity", 0)
                .remove();
        }


        this.clusterClicked = function(d, node) {
            cls = d3.select(this.view).selectAll('.cluster');

            // first, deal with selection ranges
            if (d3.event.shiftKey) {
                var firstSelectedIndex, lastSelectedIndex, currentIndex;

                cls.each(function(dl, i) {
                    if (dl.selected) {
                        firstSelectedIndex || (firstSelectedIndex = i);
                        lastSelectedIndex = i;
                    }
                    if (this === node) currentIndex = i;
                });
                var min = Math.min(firstSelectedIndex, lastSelectedIndex, currentIndex);
                var max = Math.max(firstSelectedIndex, lastSelectedIndex, currentIndex);

                // select all between first and last selected
                // when clicked inside a selection
                cls.each(function(d, i) {

                    // preserve state for additive selection
                    d.selected = ((d3.event.ctrlKey || d3.event.metaKey) && d._selected) || (i >= min && i <= max);
                });
            }
            else
            {
                // additive select with `ctrl` key
                if (!(d3.event.ctrlKey || d3.event.metaKey)) {
                    cls.each(function(d) { d.selected = false; });
                }
                d.selected = !d.selected;
            }
        }

    } // end class D3ClusterWidget

    return { 'D3ClusterWidget' : D3ClusterWidget };
});
