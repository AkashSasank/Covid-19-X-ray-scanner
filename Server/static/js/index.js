
$('.buttons-container').hover(function(){
$('.background-div').css('opacity', '0.5')
},
function(){
$('.background-div').css('opacity', '1')
})

$('#upload').hover(function(){
$('.background-div').css('opacity', '0.5');
$('.img-box').css('opacity', '1')
},
function(){
$('.background-div').css('opacity', '1');
$('.img-box').css('opacity', '0.8');
})

$('.submit').hover(function(){
$('.background-div').css('opacity', '0.5');
$('.img-box').css('opacity', '1')
},
function(){
$('.background-div').css('opacity', '1');
$('.img-box').css('opacity', '0.9');
})

$('#result-container').hover(function(){
$('.background-div').css('opacity', '0.5');
$('#result-container').css('opacity', '1')
},
function(){
$('.background-div').css('opacity', '1');
$('#result-container').css('opacity', '0.8');
})

$('.image-div').hover(function(){
$('.background-div').css('opacity', '0.5');
$('.img-box').css('opacity', '1')
},
function(){
$('.background-div').css('opacity', '1');
$('.img-box').css('opacity', '0.8');
})

$('input').on('change',function(){
  document.getElementById('preview').src = window.URL.createObjectURL(this.files[0]);
  $('#preview').css('display', 'block');
  $('label').css('display', 'none')
})
