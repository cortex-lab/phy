// ----------------------------------------------------------------------------
// Widgets
// ----------------------------------------------------------------------------

define(function(require) {
    widget = require('widgets/js/widget');
    manager = require('widgets/js/manager');
    clusterwidget = require('/nbextensions/phy/static/d3clusterwidget.js');
    require('/nbextensions/phy/static/utils.js');

    // Utility functions
    // ------------------------------------------------------------------------
    function _float_to_uint8(x) {
        // Convert a float in [0, 1] into a uint8 in [0, 255].
        return Math.floor(x * 255);
    }

    function _parse_color(color) {
        // Convert an array of float rgb values into a CSS string 'rgba(...)'.
        if (Array.isArray(color)) {
            var r = _float_to_uint8(color[0]).toString();
            var g = _float_to_uint8(color[1]).toString();
            var b = _float_to_uint8(color[2]).toString();
            return 'rgba({0}, {1}, {2}, 1)'.format(r, g, b);
        }
        return color;
    }

    // Cluster view
    // ------------------------------------------------------------------------
    var ClusterWidget = IPython.DOMWidgetView.extend({
        render: function(){
            var that = this;
            this.$el.addClass('cluster-container');
            this.mydiv = $("<div id='cv' style='width: 100%; background-color: #ede;'/>");
            this.mydiv.appendTo(this.$el);

            this.clusterd3 = new clusterwidget.D3ClusterWidget(this.mydiv[0]);

            this._clusters = [];

            this.model.on('change:clusters',
                          this.clusters_changed, this);
            this.clusters_changed();

            this.clusterd3.redraw(this.model.get('clusters'));

            this.model.on('change:colors',
                          this.colors_changed, this);
            this.colors_changed();
        },

        add_cluster: function(i) {
            //return;
            var that = this;
            console.log("<button/>");
            var cluster = $('<button>' + i.id.toString() + '</button>');
            cluster.addClass('phy-clusterview-cluster');
            //cluster.click(function () {
            //    that.model.set('value', [parseInt(i)]);
            //    that.touch();
            //});
            this._clusters.push(cluster);
            this.$el.append(cluster);
            this.clusterd3.redraw(this.model.get('clusters'));
        },

        clusters_changed: function() {
            var clusters = this.model.get('clusters');
            this.$el.find(".phy-clusterview-cluster").remove(); //empty();
            this._clusters = [];
            for (var i = 0; i < clusters.length; i++) {
                this.add_cluster(clusters[i]);
            }
            this.clusterd3.redraw(this.model.get('clusters'));
        },

        colors_changed: function() {
            var colors = this.model.get('colors');
            for (var i = 0; i < colors.length; i++) {
                var color = _parse_color(colors[i]);
                //this._clusters[i].css('background-color', color);
             }
        }
    });

    console.log("### registering the view");
    manager.WidgetManager.register_widget_view('ClusterWidget',
                                               ClusterWidget);

    return { 'ClusterWidget' : ClusterWidget };
});
