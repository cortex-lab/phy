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

Table.prototype.select = function(ids) {
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

    emit("select", ids);
};

Table.prototype.clear = function() {
    this.selected = [];
};

Table.prototype.next = function() {
    // TODO: what to do when doing next() while several items are selected.
    var id = this.selected[0];
    if (id === undefined) {
        var row = null;
        var i0 = 1;  // 1, not 0, because we skip the header.
    }
    else {
        var row = this.rows[id];
        var i0 = row.rowIndex + 1;
    }
    for (var i = i0; i < this.el.rows.length; i++) {
        row = this.el.rows[i];
        if (row.dataset.skip != 'true') {
            this.select([row.dataset.id]);
            return;
        }
    }
};

Table.prototype.previous = function() {
    // TODO: what to do when doing next() while several items are selected.
    var id = this.selected[0];
    if (id === undefined) {
        var row = null;
        var i0 = this.rows.length - 1;
    }
    else {
        var row = this.rows[id];
        var i0 = row.rowIndex - 1;
    }

    // NOTE: i >= 1 because we skip the header column.
    for (var i = i0; i >= 1; i--) {
        row = this.el.rows[i];
        if (row.dataset.skip != 'true') {
            this.select([row.dataset.id]);
            return;
        }
    }
};
