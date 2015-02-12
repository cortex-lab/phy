// main entry point for loading a session

define(function(require) {
    "use strict";

    var IPython = require('base/js/namespace');
    var session = require('services/sessions/session');
    var $ = require('jquery');
    var events = require('base/js/events');
    var manager = require('widgets/js/manager');
    var codecell = require('notebook/js/codecell');
    var configmod = require('services/config');

    console.log("loading ... in progress");





    //fake notebook because session API rely on notebook
    //TODO: remove the need to rely on Notebook in session
    var FakeNotebook = function() {
        this.events = events;              //session ctor
        this.keyboard_manager = undefined; //widget_manager ctor

        //needed to display widget
        FakeNotebook.prototype.get_msg_cell = function (msg_id) {
            return codecell.CodeCell.msg_cells[msg_id] || null;
        };

        //needed to display widget
        FakeNotebook.prototype.find_cell_index = function () {
            return 42;
        }
    }

    var HackMe = function(nbname, nbpath) {
        this.notebook_name = nbname;
        this.notebook_path = nbpath;
        this.base_url = '/';
        this.ws_url = undefined;

        this.events = events;


        function genlog(msg) {
            return function logme() {
                console.log(msg, " (args:", arguments, ")");
            }
        };

        HackMe.prototype.start = function (placeholder) {

            this.notebook = new FakeNotebook();

            var options = {
                base_url: this.base_url,
                ws_url: this.ws_url,
                notebook_path: this.notebook_path,
                notebook_name: this.notebook_name,
                kernel_name: 'python2',
                notebook: this.notebook //TODO: remove
            };


            this.session = new session.Session(options);
            this.session.start(this._onStarted.bind(this, placeholder), genlog("fail"));
        };

        HackMe.prototype._onStarted = function(placeholder) {

            this.widget_manager = this.session.kernel.widget_manager;

            //this.widget_manage
            //var comm_manager =
            //this.widget_manager = new manager.WidgetManager(comm_manager, this.notebook);

            // //TODO: remove
            // var pager = new pager.Pager('div#pager', {events: events});
            // //TODO: remove
            // var acts = new actions.init();
            // //why does it depends on pagers?
            // var keyboard_manager = new keyboardmanager.KeyboardManager({
            //     pager: pager,
            //     events: events,
            //     actions: acts });

                            //I Guess it's needed for codecell.config...
            var common_options = {
                base_url: this.base_url,
                ws_url: this.ws_url,
                notebook_path: this.notebook_path,
                notebook_name: this.notebook_name,
            };
            var config_section = new configmod.ConfigSection('notebook', common_options);
            config_section.load();

            var options = {
                events: this.events,
                config: config_section,
                keyboard_manager: undefined,
                notebook: this.notebook,
                tooltip: undefined
            }

            var cc = new codecell.CodeCell(this.session.kernel, options);
            var btn = $('<input type="button" id="field"/>');
            btn.on("click", function() {
                console.log('clicked');
                cc.execute();
            });
            placeholder.append(btn);
            placeholder.append(cc.element);
        }

    } //end class
    console.log("loading ... done");

    return { 'HackMe' : HackMe };
});
