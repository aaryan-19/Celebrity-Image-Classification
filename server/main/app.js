// // window.Dropzone.autoDiscover = false;

//     let dz = new Dropzone("#dropzone", {
//         url: "/",
//         maxFiles: 1,
//         addRemoveLinks: true,
//         dictDefaultMessage: "Some Message",
//         autoProcessQueue: false
//     });

//     dz.on("addedfile", function() {
//         if (dz.files[1]!=null) {
//             dz.removeFile(dz.files[0]);
//         }
//     });

//     dz.on("complete", function (file) {
//         // let imageData = file.dataURL;
//         console.log(file.dataURL)
//         var url = "http://127.0.0.1:3000/classify";

//         $.post(url, {
//             image_data: file.dataURL
//         },function(data, status) {
//             console.log(data)
//             window.location.href = "/pagemaster?name="+data["celeb"]+"&p="+data['probability'];

//             // dz.removeFile(file);
//         });
//     });

//     $("#submitBtn").on('click', function (e) {
//         dz.processQueue();
//     });



// //

$(document).ready(function() {
    // Select the form and input elements
    var form = $('#myform');
    var fileInput = $('#image');

    // When the form is submitted
    form.submit(function(event) {
      event.preventDefault(); // Prevent the default form submission

      // Create a new FormData object
      var formData = new FormData();

      // Append the selected file to the FormData object
      formData.append('image', fileInput[0].files[0]);

      // Send the FormData object to the server using jQuery's ajax method
      $.ajax({
        url: "http://127.0.0.1:3000/classify", // Replace with your server's URL
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function(response) {
          console.log('File uploaded successfully');
          console.log(response); // Log the server's response
          data=response[0]

          document.querySelector("body").classList.add("hidden")
          window.location.href = "http://127.0.0.1:3000/pagemaster?name="+data["celeb"]+"&p="+data['probability'];
        },
        error: function(jqXHR, textStatus, errorThrown) {
          console.error('Error uploading file: ' + textStatus);
        }
      });
    });
  });
