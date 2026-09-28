import io
import unittest
import numpy as np
import torch
from core.lppm.value_families import FAMILIES,EXTRA_FAMILIES,ValueFamily,predict_family


class ValueFamilyTests(unittest.TestCase):
    def test_nonnegative_finite_gradients_for_every_family(self):
        for family in FAMILIES + EXTRA_FAMILIES:
            with self.subTest(family=family):
                net=ValueFamily(family,5,8)
                x=torch.randn(2,4,5,requires_grad=True)
                q=torch.tensor([[0.,0.,1.,1.],[0.,0.,0.,1.]])
                v=net(x,q)
                self.assertEqual(v.shape,q.shape)
                self.assertTrue(torch.isfinite(v).all() and (v>=0).all())
                self.assertTrue((v[q.bool()]>.01).all())
                v.sum().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertGreater(net.accepting_raw.grad.item(),0)

    def test_serialization_and_batch_independence(self):
        for family in FAMILIES + EXTRA_FAMILIES:
            with self.subTest(family=family):
                net=ValueFamily(family,5,8,torch.randn(8,5) if family=='rbf' else None)
                model=dict(family=family,width=8,mean=np.zeros(5),scale=np.ones(5),weights=net.state_dict())
                z=np.random.default_rng(7).normal(size=(2,4,5))
                d=np.array([[0,0,1,1],[0,0,0,1]],dtype=bool)
                v=predict_family(model,z,d)
                single=predict_family(model,z[:1],d[:1])
                np.testing.assert_allclose(v[:1],single,atol=1e-6)
                buf=io.BytesIO(); torch.save(model,buf); buf.seek(0)
                reloaded=torch.load(buf,weights_only=False)
                np.testing.assert_array_equal(v,predict_family(reloaded,z,d))

    def test_centers_and_frequencies_are_not_trainable(self):
        for family,buffer in [('rbf','centers'),('fourier','frequencies')]:
            net=ValueFamily(family,5,8)
            self.assertIn(buffer,dict(net.named_buffers()))
            self.assertNotIn(buffer,dict(net.named_parameters()))

    def test_invalid_schema_fails(self):
        with self.assertRaises(ValueError): ValueFamily('unknown',5)
        with self.assertRaises(ValueError): ValueFamily('rbf',5,8,torch.zeros(4,5))
        with self.assertRaises(ValueError): ValueFamily('linear',5)(torch.zeros(2,3,5),torch.zeros(2,2))


if __name__=='__main__': unittest.main()
