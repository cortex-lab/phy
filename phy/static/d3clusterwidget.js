
define(function(require) {
    d3 = require('/nbextensions/phy/static/d3.js');

    function D3ClusterWidget(clusterView) {
        this.view = clusterView;
        this.width = 300;
        this.height = 300;

        this.redraw = function(clusters)
        {
            var width = 300;
            var height = 300;
            var barWidth = 42;

            //var clusters = generate_all_clusters();

            var clusterView = d3.select(this.view);
            var cluster = clusterView.selectAll(".cluster")
                .data(clusters, function(d) { return d.id; });

            //
            // Clusters entering the view
            //

            var clusterEnter = cluster.enter()
                .insert("div")
                .attr("class", "cluster")
                .style("display", "inline-block")
                .style("margin",  "5px")
                .style("padding-top",  "10px")
                .style("background-color",  "white")
                .style("border-radius",  "4px")
                .style("border",  "3px solid black")
                .style("text-align",  "center")
                .style("width", "130px")
                .style("fill-opacity", 0);

            var lineFunction = d3.svg.line()
                .x(function(d, i) { return width-(i*barWidth); })
                .y(function(d, i) { return height-d; })
                .interpolate("step-after");

            var svgContainer = clusterEnter.append("svg")
            .attr("width", width)
            .attr("height", height);


            svgContainer.append("path")
                .datum(function(d) {barWidth = width/d.ccg.length; return d.ccg;})
                .attr("d", function(d) {return lineFunction(d)})
                .attr("stroke", "black")
                .attr("stroke-width", 0)
                .attr("fill", "black")

            clusterEnter.append("div")
                .text(function(d) { return d.id; });

            clusterEnter.append("div")
                .text(function(d) { return d.nchannels + "ch"; });

            //
            // Clusters to update
            //

            var clusterUpdate = d3.transition(cluster)
                .style("fill-opacity", 1);

            clusterUpdate.select(".cluster");

            //
            // Clusters to remove
            //

            var clusterExit = d3.transition(cluster.exit())
                .style("fill-opacity", 0)
                .remove();

            clusterExit.select(".cluster");
        }
    }
    return { 'D3ClusterWidget' : D3ClusterWidget };
});
