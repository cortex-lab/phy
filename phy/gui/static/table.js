// Utils.

function uniq(a) {
    var seen = {};
    return a.filter(function(item) {
        return seen.hasOwnProperty(item) ? false : (seen[item] = true);
    });
}

function isFloat(n) {
    return n === Number(n) && n % 1 !== 0;
}


// Table class.

var Table = function (el) {
    this.el = el;
    this.selected = [];
    this.headers = {};  // {name: th} mapping
    this.rows = {};  // {id: tr} mapping
    this.tablesort = null;
};

Table.prototype.setData = function(data) {
    /*
    data.cols: list of column names
    data.items: list of rows (each row is an object {col: value})
     */
    if (data.items.length == 0) return;

    // Reinitialize the state.
    this.selected = [];
    this.rows = {};

    var that = this;
    var keys = data.cols;

    // Clear the table.
    while (this.el.firstChild) {
        this.el.removeChild(this.el.firstChild);
    }

    var thead = document.createElement("thead");
    var tbody = document.createElement("tbody");

    // Header.
    var tr = document.createElement("tr");
    for (var j = 0; j < keys.length; j++) {
        var key = keys[j];
        var th = document.createElement("th");
        th.appendChild(document.createTextNode(key));
        tr.appendChild(th);
        this.headers[key] = th;
    }
    thead.appendChild(tr);

    // Data rows.
    for (var i = 0; i < data.items.length; i++) {
        tr = document.createElement("tr");
        var row = data.items[i];
        for (var j = 0; j < keys.length; j++) {
            var key = keys[j];
            var value = row[key];
            // Format numbers.
            if (isFloat(value))
                value = value.toPrecision(3);
            var td = document.createElement("td");
            td.appendChild(document.createTextNode(value));
            tr.appendChild(td);
        }

        // Set the data values on the row.
        for (var key in row) {
            tr.dataset[key] = row[key];
        }

        tr.onclick = function(e) {
            var id = parseInt(String(this.dataset.id));
            var evt = e ? e:window.event;
            // Control pressed: toggle selected.
            if (evt.ctrlKey || evt.metaKey) {
                var index = that.selected.indexOf(id);
                // If this item is already selected, deselect it.
                if (index != -1) {
                    var selected = that.selected.slice();
                    selected.splice(index, 1);
                    that.select(selected);
                }
                // Otherwise, select it.
                else {
                    that.select(that.selected.concat([id]));
                }
            }
            // Otherwise, select just that item.
            else {
                that.select([id]);
            }
        }

        tbody.appendChild(tr);
        this.rows[data.items[i].id] = tr;
    }

    this.el.appendChild(thead);
    this.el.appendChild(tbody);

    // Enable the tablesort plugin.
    this.tablesort = new Tablesort(this.el);
};

Table.prototype.sortBy = function(header, dir) {
    dir = typeof dir !== 'undefined' ? dir : 'asc';
    if (this.headers[header] == undefined)
        throw "The column `" + header + "` doesn't exist."
    this.tablesort.sortTable(this.headers[header]);
    if (dir == 'desc') {
        this.tablesort.sortTable(this.headers[header]);
    }
};

Table.prototype.currentSort = function() {
    for (var header in this.headers) {
        if (this.headers[header].classList.contains('sort-up')) {
            return [header, 'desc'];
        }
        if (this.headers[header].classList.contains('sort-down')) {
            return [header, 'asc'];
        }
    }
    return [null, null];
};

Table.prototype.select = function(ids, do_emit) {
    do_emit = typeof do_emit !== 'undefined' ? do_emit : true;

    ids = uniq(ids);

    // Remove the class on all rows.
    for (var i = 0; i < this.selected.length; i++) {
        var id = this.selected[i];
        var row = this.rows[id];
        row.classList.remove('selected');
    }

    // Add the class.
    for (var i = 0; i < ids.length; i++) {
        ids[i] = parseInt(String(ids[i]));
        this.rows[ids[i]].classList.add('selected');
    }

    this.selected = ids;

    if (do_emit)
        emit("select", ids);
};

Table.prototype.clear = function() {
    this.selected = [];
};

Table.prototype.firstRow = function() {
    return this.el.rows[1];
};

Table.prototype.lastRow = function() {
    return this.el.rows[this.el.rows.length - 1];
};

Table.prototype.rowIterator = function(id) {
    // TODO: what to do when doing next() while several items are selected.
    var i0 = 1;
    if (id !== undefined) {
        i0 = this.rows[id].rowIndex;
    }
    var that = this;
    return {
        i: i0,
        n: that.el.rows.length,
        row: function () { return that.el.rows[this.i]; },
        previous: function () {
            if (this.i > 1) {
                this.i--;
            }
            return this.row();
        },
        next: function () {
            if (this.i < this.n - 1) {
                this.i++;
            }
            return this.row();
        }
    };
};

Table.prototype.next = function() {
    // TODO: what to do when doing next() while several items are selected.
    var id = this.selected[0];
    if (id == undefined) {
        var row = this.firstRow();
    }
    else {
        // Select the next non-skip.
        var iterator = this.rowIterator(id);
        var row = iterator.next();
        while (row.dataset.skip == 'true') {
            row = iterator.next();
        }
    }
    this.select([row.dataset.id]);
    row.scrollIntoView(false);
    return;
};

Table.prototype.previous = function() {
    // TODO: what to do when doing previous() while several items are selected.
    var id = this.selected[0];
    if (id == undefined) {
        var row = this.lastRow();
    }
    else {
        // Select the previous non-skip.
        var iterator = this.rowIterator(id);
        var row = iterator.previous();
        while (row.dataset.skip == 'true') {
            row = iterator.previous();
        }
    }
    this.select([row.dataset.id]);
    row.scrollIntoView(false);
    return;
};
