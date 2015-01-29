// ----------------------------------------------------------------------------
// Widgets
// ----------------------------------------------------------------------------

require(["widgets/js/widget", "widgets/js/manager"],
        function(widget, manager) {

    // Cluster view
    // ------------------------------------------------------------------------
    var ClusterWidget = IPython.DOMWidgetView.extend({
        render: function(){
            var that = this;
            this.$el.addClass('cluster-container');
            this._clusters = [];

            this.model.on('change:clusters',
                          this.clusters_changed, this);
            this.clusters_changed();

            this.model.on('change:colors',
                          this.colors_changed, this);
            this.colors_changed();
        },

        add_cluster: function(i) {
            var that = this;
            var cluster = $('<button>' + i.toString() + '</div>');
            cluster.addClass('phy-clusterview-cluster');
            cluster.click(function () {
                that.model.set('value', [parseInt(i)]);
                that.touch();
            });
            this._clusters.push(cluster);
            this.$el.append(cluster);
        },

        clusters_changed: function() {
            var clusters = this.model.get('clusters');
            this.$el.empty();
            this._clusters = [];
            for (var i = 0; i < clusters.length; i++) {
                this.add_cluster(clusters[i]);
            }
        },

        colors_changed: function() {
            var colors = this.model.get('colors');
            for (var i = 0; i < colors.length; i++) {
                var color = _parse_color(colors[i]);
                console.log(color);
                this._clusters[i].css('background-color', color);
            }
        }
    });

    manager.WidgetManager.register_widget_view('ClusterWidget',
                                               ClusterWidget);

});
