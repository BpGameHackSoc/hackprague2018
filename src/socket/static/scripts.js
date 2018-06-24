var latest = document.getElementById('latest');

var xhr = new XMLHttpRequest();

count = 0
var label_names = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutrality']

xhr.onreadystatechange = function() {

    function handleNewData() {
        var messages = xhr.responseText.split('\n');
        
        messages.slice(count, -1).forEach(function(value) {
            str_values = value.split(',')
            for (i=0;i<str_values.length;i++) {
                v = parseFloat(str_values[i])
                id = '#'+label_names[i]+'_v'
                $(id).val(v)
            }
            
            //latest.textContent = value;  // update the latest value in place
        });

        count = messages.length-1
    }

    var timer;

    timer = setInterval(function() {
        // check the response for new data
        handleNewData();
        // stop checking once the response has ended
        if (xhr.readyState == XMLHttpRequest.DONE) {
            clearInterval(timer);
            latest.textContent = 'Done';
        }
    }, 1000);

}


xhr.open('GET', "stream");
xhr.send();