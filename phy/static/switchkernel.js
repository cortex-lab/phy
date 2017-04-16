define(function(require) {
    IPython = require('base/js/namespace');
    utils = require('base/js/utils');

    // TO Insert into a notebook:
    // %%javascript
    // require(['/nbextensions/phy/static/switchkernel.js']);
    // var sk = require('/nbextensions/phy/static/switchkernel.js');
    // console.log(sk);
    // sk.switchKernel('proto/proto_kwik.ipynb');

    //switch the kernel of the current notebook.
    // (take a notebook path like : myfolder/mynotebook.ipynb)
    var switchKernel= function(notebookpath) {
        var mydata = IPython.notebook.session._get_model();
        mydata.kernel.id = '';
        mydata.notebook.path = notebookpath;
        //get the kernel id of another notebook
        $.ajax(IPython.notebook.session.session_service_url, {
            processData: false,
            cache: false,
            type: "POST",
            data: JSON.stringify(mydata),
            dataType: "json",
            success: function(data) {
                console.log("kernel:", data.kernel.id);
                var mykernelId = data.kernel.id
                //Inject code into Kernel to force the kernel ID
                require('services/kernels/kernel').Kernel.prototype._kernel_created = function (data) {
                    this.id = mykernelId;
                    this.kernel_url = utils.url_join_encode(this.kernel_service_url, this.id);
                    this.start_channels();
                };
                IPython.notebook.session.restart();
            },
            error: function() { console.log("arguments:", arguments); }
        });
    }

    return { 'switchKernel' : switchKernel };
});
//switchKernel('proto/HackMyKernel.ipynb');
//switchKernel('proto/proto_kwik.ipynb');
