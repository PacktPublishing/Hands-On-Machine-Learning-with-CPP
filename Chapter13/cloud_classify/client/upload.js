const url = 'http://localhost:8080/multipart';
const form = document.querySelector('form');

form.addEventListener('submit', e => {
    e.preventDefault();

    const files = document.querySelector('[type=file]').files;
    const formData = new FormData();

    for (let i = 0; i < files.length; i++) {
        let file = files[i];

        formData.append('files[]', file);
    }

    fetch(url, {
        method: 'POST',
        body: formData
    }).then(response => {
        console.log(response);
        if (response.ok) {
          let text = response.text();
          text.then(data=>{
            alert(data)});
        } else {
          alert("Error :" + response.status);
        }

        //document.write(response);
    });
});
