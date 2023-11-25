
export class Model {
    id: string
    name: string

    constructor(data: any) {
        this.id = data.id
        this.name = data.name
    }

    toString() {
        return this.name
    }
}

export class ModelResult {
    result: string

    constructor(data: any) {
        this.result = data.result
    }
}

export class SendFrameResponse {
    got_video: boolean
    id: string

    constructor(data: any) {
        this.got_video = data.got_video
        this.id = data.id
    }
}


export async function getModels(): Promise<Model[]> {
    let result = await fetch("http://localhost:8000/api/models/", {
        method: "GET",
    })

    return (await result.json()).map((x) => new Model(x))
}

// export async function sendFile(file): Promise<boolean> {
//     // TODO: populate me plz
//     // Don't forget about types as well!
//     let result = await fetch("http://localhost:8000/api/media-file/", {
//         method: "POST",
//     })
// }

function dataURLtoFile(dataurl: string, filename: string) {
    var arr = dataurl.split(','),
        mime = arr[0].match(/:(.*?);/)[1],
        bstr = atob(arr[arr.length - 1]),
        n = bstr.length,
        u8arr = new Uint8Array(n);
    while (n--) {
        u8arr[n] = bstr.charCodeAt(n);
    }
    return new File([u8arr], filename, { type: mime });
}


export async function sendFrame(frameData: string): Promise<SendFrameResponse | undefined> {
    const formData = new FormData();
    formData.append('file', dataURLtoFile(frameData, "image.jpg"));

    const response = await fetch('http://localhost:8000/api/media-file/', {
        method: 'POST',
        body: formData,
    });
    if (response.ok) {
        const result = await response.json();
        return new SendFrameResponse(result);
    } else {
        console.error('Error sending frame:', response.status);
    }
}

// let fileVar = null;
// let fileName;

// export  async function post_form(){
//     let dataArray = new FormData();
//     dataArray.append("imagefileinput", fileVar[0], fileName);
//     fetch("/post", {
//       method: "POST",
//       headers: {
//           'Content-Type': 'multipart/form-data'
//         },
//       body: dataArray
//     })
//     .then(response => {
//       console.log(response);
//     })
//     .catch(error => {
//       console.log(error);
//     });
//   }

export async function getResult(id: string, model: string): Promise<ModelResult> {
    let result = await fetch(`http://localhost:8000/api/process/?id=${id}&model=${model}`, {
        method: "POST",
    })

    return new ModelResult(await result.json())
}