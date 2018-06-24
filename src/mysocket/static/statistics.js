window.onload = function () {



CanvasJS.addColorSet("emoticonColors",
    [
    "#F00",
    "#090",
    "#00B",
    "#FF0",
    "#A0A",
    "#F90",
    "#FFF"               
    ]
);

//Better to construct options first and then pass it as a parameter
var options = {
    title: {
        text: "Emotions recognized"              
    },
    data: [              
    {
        // Change type to "doughnut", "line", "splineArea", etc.
        type: "column",
        dataPoints: []
    }
    ]
};

$("#chartContainer").CanvasJSChart(options);

var label_names = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutrality']
var count = label_names.length

var dps = []
for (i=0; i<count; i++){
    dps.push({label: label_names[i], y: 1/count})
}

var dps_real = [0, 0, 0, 0, 0, 0, 0]

var updateInterval = 1000;

var chart = new CanvasJS.Chart("chartContainer", {
    backgroundColor: "#113",
    colorSet: "emoticonColors",
    // title :{
    //     text: "Emotions recognized",
    //     fontColor: "#FFF",
    // },      
    data: [{
        type: "column",
        dataPoints: dps,

    }],
    axisY: {
        minimum: 0,
        maximum: 1,
        interval: 0.2,
        labelFontColor: "#999",
    },
    axisX: {
        labelFontColor: "#999",
    }

});


var updateChart = function () {

    // s = 0
    // for (var j = 0; j < count; j++) {
    //     dps_real[j] += distribution[j]
    //     s += dps_real[j]
    // }

    for (var j = 0; j < count; j++) {
        v = parseFloat($('#'+label_names[j] + '_v').val())
        dps[j] = {label : label_names[j], y: v}
    }

    console.log(dps)


    chart.render();
};

function generate_random_numbers(n) {
    a = []
    s = 0
    for (i = 0; i < n; i++) {
        r = Math.random()
        a.push(r)
        s += r
    }

    for (i = 0; i < n; i++) {
        a[i] /= s
    }

    return a
}

function resize(id, shift, callback) {
    $(id).animate({
        width: $(id).width() + shift,
        height: $(id).height() + shift},
    updateInterval/2);
}


(function pulse(start, close_id) {

    size_s = 32
    size_l = 64
    opacity_s = 0.8
    opacity_l = 1.


    if (!start) {
        $(close_id).animate({
            width:  size_s,
            height: size_s,
            opacity: opacity_s}, updateInterval/2);
        setTimeout(function(){}, updateInterval/2);
    }

    winning_serial = 0
    winning_value = $('#'+label_names[winning_serial] + '_v').val()

    for (i=1;i<count;i++) {
        new_value =  $('#'+label_names[i] + '_v').val()
        if (new_value > winning_value) {
            winning_serial = i
            winning_value = $('#'+label_names[winning_serial] + '_v').val()
        }
    }

    winning_id = '#'+label_names[winning_serial]



    $(winning_id).animate({
        width:  size_l,
        height: size_l,
        opacity: opacity_l}, updateInterval/2, function(){pulse(false, winning_id)}
    );

    // $(winning_id).animate({
    //     'font-size': (back) ? '100px'
    //                         : '140px',
    //     opacity: (back) ? 1 : 0.8
    // }, updateInterval/2, function(){pulse(!back)});

})(true, 'asd');




updateChart(generate_random_numbers(count));
setInterval(function(){updateChart()}, updateInterval);



}