
function print_file_date(url) {
    try {
        var req = new XMLHttpRequest();
        req.open("HEAD", url, false);
        req.send(null);
        if(req.status == 200){
            var lastModDate = new Date(req.getResponseHeader('Last-Modified'))
            return "<p><i>Last update: " + lastModDate.toLocaleDateString() + "</i></p>";
        }
        else return ""; //false;
    } catch(er) {
        return ""; //er.message;
    }
}

