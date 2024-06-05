
(function($) {
	$(document).ready(function () { 
		$('.heading').click(function() { 
			var heading = $(this);
			var content = $(this).next();
			if(content.css('display') == 'none'){
				heading.find('.icon-close-open').css('background-position', '0px -20px');
				content.css('display', 'block');
			}
			else{
				heading.find('.icon-close-open').css('background-position', '0px 0px');
				content.css('display', 'none'); 
			}
		});
	});
})(jQuery);

