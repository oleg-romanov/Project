using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering.PostProcessing;

public class DepthOfFieldController : MonoBehaviour
{
    Ray raycast;
    RaycastHit hit;
    bool isHit;
    float hitDistance;

    public PostProcessVolume volume;

    DepthOfField depthOfField;

    // Start is called before the first frame update
    void Start()
    {
        volume.profile.TryGetSettings(out depthOfField);
    }

    // Update is called once per frame
    void Update()
    {
        raycast = new Ray(transform.position, transform.forward * 100);

        isHit = false;

        if (Physics.Raycast(raycast, out hit, 100f))
        {
            isHit = true;
            hitDistance = Vector3.Distance(transform.position, hit.point);
            Debug.Log("Hit");
        }
        else
        {
            if(hitDistance < 100f)
            {
                hitDistance++;
            }
        }

        SetFocus();
    }

    void SetFocus()
    {
        depthOfField.focusDistance.value = hitDistance;
        // depthOfField.focusDistance.value = Mathf.Lerp(depthOfField.focusDistance.value, hitDistance, Time.deltaTime * focusSpeed);
    }


    private void OnDrawGizmos()
    {
        if (isHit)
        {
            Gizmos.DrawSphere(hit.point, 0.1f);

            Debug.DrawRay(transform.position, transform.forward * Vector3.Distance(transform.position, hit.point));
        }
        else
        {
            Debug.DrawRay(transform.position, transform.forward * 100f);
        }
    }
}
