
var Table = function (el) {
    this.el = el;
    this.selected = [];
    this.headers = {};  // {name: th} mapping
    this.rows = {};  // {id: tr} mapping
    this.tablesort = null;
};

Table.prototype.setData = function(data) {
    if (data.items.length == 0) return;
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
            var td = document.createElement("td");
            td.appendChild(document.createTextNode(value));
            tr.appendChild(td);
        }

        // Set the data values on the row.
        for (var key in row) {
            tr.dataset[key] = row[key];
        }

        tr.onclick = function(e) {
            var selected = [this.dataset.id];

            var evt = e ? e:window.event;
            if (evt.ctrlKey || evt.metaKey) {
                selected = that.selected.concat(selected);
            }
            that.select(selected);
        }

        tbody.appendChild(tr);
        this.rows[data.items[i].id] = tr;
    }

    this.el.appendChild(thead);
    this.el.appendChild(tbody);

    // Enable the tablesort plugin.
    this.tablesort = new Tablesort(this.el);
};

Table.prototype.sortBy = function(header) {
    this.tablesort.sortTable(this.headers[header]);
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

Table.prototype.select = function(ids, raise_event) {
    // The default is true.
    raise_event = typeof raise_event !== 'undefined' ? false : true;

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

    if (raise_event) {
        emit("select", ids);
    }
};

Table.prototype.clear = function() {
    this.selected = [];
};

Table.prototype.next = function() {
    // TODO: what to do when doing next() while several items are selected.
    var id = this.selected[0];
    var row = this.rows[id];
    var i0 = row.rowIndex + 1;
    var items = [];

    for (var i = i0; i < this.el.rows.length; i++) {
        row = this.el.rows[i];
        if (!(row.dataset.skip)) {
            items.push(row.dataset.id);
            break;
        }
    }

    if (!(items.length)) return;

    this.select(items);
};

Table.prototype.previous = function() {
    // TODO: what to do when doing next() while several items are selected.
    var id = this.selected[0];
    var row = this.rows[id];
    var i0 = row.rowIndex - 1;
    var items = [];

    // NOTE: i >= 1 because we skip the header column.
    for (var i = i0; i >= 1; i--) {
        row = this.el.rows[i];
        if (!(row.dataset.skip)) {
            items.push(row.dataset.id);
            break;
        }
    }

    if (!(items.length)) return;

    this.select(items);
};
